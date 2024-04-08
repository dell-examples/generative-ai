// Created by Scalers AI for Dell Inc.

package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/gin-gonic/gin"
)

type Data struct {
	LocalDescription string
}

var browserDescription string
var localDescription string
var data Data

func main() {
	r := gin.Default()
	browserDescription = "no offer"
	localDescription = "no offer"
	r.LoadHTMLGlob("*.html")
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "webrtc.html", nil)
	})

	r.PUT("/setBrowserDescription", func(c *gin.Context) {
		jsonData, err := ioutil.ReadAll(c.Request.Body)
		str1 := bytes.NewBuffer(jsonData).String()
		browserDescription = base64.StdEncoding.EncodeToString([]byte(str1))
		if err != nil {
			fmt.Print(err)
		}
		c.JSON(http.StatusOK, gin.H{"data": "setBrowserDescription"})
	})

	r.PUT("/setLocalDescription", func(c *gin.Context) {
		jsonData, err := ioutil.ReadAll(c.Request.Body)
		str1 := bytes.NewBuffer(jsonData).String()
		Data := []byte(str1)
		json.Unmarshal(Data, &data)
		println(data.LocalDescription)
		localDescription = data.LocalDescription
		if err != nil {
			fmt.Print(err)
		}
		c.JSON(http.StatusOK, gin.H{"data": "setLocalDescription"})
	})

	r.GET("/getBrowserDescription", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"browserDescription": browserDescription})
		browserDescription = "no offer"
	})

	r.GET("/getLocalDescription", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"localDescription": localDescription})
		localDescription = "no offer"
	})
	r.Run(":9000")
}
