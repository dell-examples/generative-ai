// Created by Scalers AI for Dell Inc.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	gst "github.com/pion/example-webrtc-applications/v3/internal/gstreamer-src"
	"github.com/pion/example-webrtc-applications/v3/internal/signal"
	"github.com/pion/webrtc/v3"
	"gopkg.in/yaml.v3"
)

type BrowserDescriptionJson struct {
	BrowserDescription string
}
type Data struct {
	LocalDescription string
}

var browserDescriptionJson BrowserDescriptionJson

func main() {

	// Prepare the configuration
	config := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{
			{
				URLs: []string{"stun:stun.l.google.com:19302"},
			},
		},
	}

	// Create a new RTCPeerConnection
	peerConnection, err := webrtc.NewPeerConnection(config)
	if err != nil {
		panic(err)
	}

	// Set the handler for ICE connection state
	// This will notify you when the peer has connected/disconnected
	peerConnection.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
		fmt.Printf("Connection State has changed %s \n", connectionState.String())
		if connectionState.String() == "disconnected" || connectionState.String() == "failed" {
			os.Exit(0)
		}
	})

	// Read pipeline config
	conf_file, err := ioutil.ReadFile("pipeline_config.yml")

	if err != nil {

		log.Fatal(err)
	}

	conf_map := make(map[interface{}]interface{})

	err2 := yaml.Unmarshal(conf_file, &conf_map)

	if err2 != nil {

		log.Fatal(err2)
	}

	// Parse pipeline config and get list of streams with visualization enabled
	for k, v := range conf_map {
		if k == "rtsp_streams" {
			streams, err := json.Marshal(v)
			if err != nil {
				fmt.Printf("Error: %s", err.Error())
			} else {
				var streams_map map[string]interface{}
				json.Unmarshal([]byte(streams), &streams_map)
				i := 0
				for _, value := range streams_map {
					stream, err := json.Marshal(value)
					if err != nil {
						fmt.Printf("Error: %s", err.Error())
					} else {
						var stream_map map[string]interface{}
						json.Unmarshal([]byte(stream), &stream_map)
						if stream_map["visualize"] == true {
							fmt.Println(stream_map["udp_port"])
							pipeline := fmt.Sprintf("udpsrc port=%v ! decodebin ! videoscale ! videoconvert ! queue", stream_map["udp_port"])
							fmt.Println(pipeline)
							VideoTrack1, err := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{MimeType: "video/vp8"}, "video", "pion"+strconv.Itoa(i))
							if err != nil {
								panic(err)
							}
							_, err = peerConnection.AddTrack(VideoTrack1)
							if err != nil {
								panic(err)
							}
							gst.CreatePipeline("vp8", []*webrtc.TrackLocalStaticSample{VideoTrack1}, pipeline).Start()
							i++

						}
					}
				}
			}
		}
	}

	for {
		response, err := http.Get("http://localhost:9000/getBrowserDescription")

		if err != nil {
			fmt.Print(err.Error())
			os.Exit(1)
		}

		responseData, err := ioutil.ReadAll(response.Body)
		if err != nil {
			log.Fatal(err)
		}
		json.Unmarshal([]byte(responseData), &browserDescriptionJson)
		if browserDescriptionJson.BrowserDescription != "no offer" {
			break
		}
		fmt.Println(browserDescriptionJson.BrowserDescription)
		time.Sleep(2 * time.Second)
	}

	// Wait for the offer to be pasted
	offer := webrtc.SessionDescription{}
	signal.Decode(browserDescriptionJson.BrowserDescription, &offer)

	// Set the remote SessionDescription
	err = peerConnection.SetRemoteDescription(offer)
	if err != nil {
		panic(err)
	}

	// Create an answer
	answer, err := peerConnection.CreateAnswer(nil)
	if err != nil {
		panic(err)
	}

	// Create channel that is blocked until ICE Gathering is complete
	gatherComplete := webrtc.GatheringCompletePromise(peerConnection)

	// Sets the LocalDescription, and starts our UDP listeners
	err = peerConnection.SetLocalDescription(answer)
	if err != nil {
		panic(err)
	}

	<-gatherComplete

	// Output the answer in base64 so we can paste it in browser
	data := Data{
		LocalDescription: signal.Encode(*peerConnection.LocalDescription()),
	}
	json, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}
	client := &http.Client{}
	req, err := http.NewRequest(http.MethodPut, "http://localhost:9000/setLocalDescription", bytes.NewBuffer(json))
	println(req)
	// set the request header Content-Type for json
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}

	fmt.Println(resp.StatusCode)

	// Block forever
	select {}
}
