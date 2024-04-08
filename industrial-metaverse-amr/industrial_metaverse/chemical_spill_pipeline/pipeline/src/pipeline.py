# Created by Scalers AI for Dell Inc.

import sys

sys.path.append("./")
import gi

gi.require_version("Gst", "1.0")
import argparse
import json
import time
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import pyds
import yaml
import zenoh
from gi.repository import GLib, Gst
from utils import bus_call


class DeepstreamSegmentation:
    def __init__(
        self, deploy_single, amr_stream_srcs, zenoh_port, udp_port, topic
    ):
        self.deploy_single = deploy_single
        self.amr_stream_srcs = amr_stream_srcs
        self.zenoh_port = zenoh_port
        self.udp_port = udp_port
        self.topic = topic

        self.overlay_alert = False
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0
        self.total_fps = 0
        self.conf_threshold = 70

        self.prev_time = time.time()
        self.alert_overlay = self.load_alert_image()

        Gst.init(None)

    def update_fps(self) -> dict:
        """Calculate pipeline FPS.

        :returns fps_details: Current and average fps details
        """
        self.frame_count += 1
        self.total_frames += 1
        inference_time = time.time() - self.prev_time

        if inference_time >= 1.0:
            self.fps = self.frame_count
            self.prev_time = time.time()
            self.frame_count = 0

        self.total_fps += self.fps
        avg_fps = self.total_fps / self.total_frames
        fps_details = {"curr_fps": self.fps, "avg_fps": round(avg_fps, 2)}

        return fps_details

    def load_alert_image(self):
        "Loads the alert image for the overlay with 4 channels."
        image_path = "/src/assets/alert.png"

        alert = cv2.imread(image_path)
        b_channel, g_channel, r_channel = cv2.split(alert)
        alert = cv2.merge((r_channel, g_channel, b_channel))

        alpha_channel = np.full((150, 150, 1), 255, dtype=np.uint8)
        overlay_resized = cv2.resize(alert, (150, 150))
        alert_overlay = np.concatenate(
            (overlay_resized, alpha_channel), axis=2
        )

        return alert_overlay

    def establish_zenoh_connection(self, broker, key, max_retries=3):
        """Establishes a connection with Zenoh broker."""
        retries = 0
        session = None
        publisher = None
        publisher_meta = None

        while retries < max_retries:
            try:
                zenoh_config = zenoh.Config()
                zenoh_config.insert_json5(
                    zenoh.config.CONNECT_KEY, json.dumps([broker])
                )
                zenoh_config.insert_json5(
                    "scouting/multicast/enabled", "false"
                )
                session = zenoh.open(zenoh_config)
                publisher = session.declare_publisher(key)
                publisher_meta = session.declare_publisher("metadata")

                return session, publisher, publisher_meta
            except Exception:
                if retries < max_retries - 1:
                    print(
                        "Retrying to get the Zenoh broker "
                        f"connection ({retries + 1}/{max_retries})"
                    )
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        print(
            "Zenoh broker connection cannot be established "
            f"{max_retries} retries. Exiting."
        )
        if session is not None:
            session.close()

        sys.exit(1)

    def disable_overlay(self, obj_meta):
        """Disable text and bounding boxes overlays."""
        obj_meta.rect_params.border_width = 0
        obj_meta.text_params.display_text = ""
        obj_meta.text_params.set_bg_clr = 0

    def overlay_pub_probe(self, pad, info, publishers=None):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                # Extract the data from the GStreamer buffer for the current frame
                frame = pyds.get_nvds_buf_surface(
                    hash(gst_buffer), frame_meta.batch_id
                )
                # overlay the alerts
                if self.overlay_alert:
                    frame[0:150, 1770:1920] = self.alert_overlay
                    self.overlay_alert = False

                frame_copy = np.array(frame, copy=True, order="C")
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
            except StopIteration:
                break

            # overlay FPS
            fps = f"FPS: {self.update_fps()['curr_fps']}"

            # streams are published over zenoh for the multi device deployment
            if not self.deploy_single:
                # overlay the fps values
                cv2.putText(
                    frame_copy,
                    fps,
                    (30, 135),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                # publish the frame over the zenoh
                image_buffer = frame_copy.astype(np.uint8).tobytes()
                publishers[frame_meta.pad_index].put(image_buffer)
            else:
                cv2.putText(
                    frame,
                    fps,
                    (30, 135),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def spill_detection_probe(self, pad, info):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                if (obj_meta.confidence * 100) > self.conf_threshold:
                    if obj_meta.class_id == 0:
                        message = f"{self.topic} detected spill"
                        self.overlay_alert = True
                        self.spill_pub.put(message)
                    else:
                        self.disable_overlay(obj_meta)
                else:
                    self.disable_overlay(obj_meta)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def cb_newpad(self, decodebin, decoder_src_pad, data):
        print("In cb_newpad\n")
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        if gstname.find("video") != -1:
            if features.contains("memory:NVMM"):
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write(
                        "Failed to link decoder src pad to source bin ghost pad\n"
                    )
            else:
                sys.stderr.write(
                    " Error: Decodebin did not pick nvidia decoder plugin.\n"
                )

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        print("Decodebin child added:", name, "\n")
        if name.find("decodebin") != -1:
            Object.connect(
                "child-added", self.decodebin_child_added, user_data
            )

        if "source" in name:
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property("drop-on-latency") is not None:
                Object.set_property("drop-on-latency", True)

    def create_source_bin(self, index, uri):
        print("Creating source bin")

        bin_name = "source-bin-%02d" % index
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        uri_decode_bin = Gst.ElementFactory.make(
            "uridecodebin", "uri-decode-bin"
        )
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)

        nbin.add(uri_decode_bin)
        bin_pad = nbin.add_pad(
            Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        )
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin

    def run(self):
        is_live = False
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        number_sources = len(self.amr_stream_srcs)

        broker = f"tcp/{amr_streams['NOVA_CARTER_B1']['broker']}:7445"
        (_, self.spill_pub, _) = self.establish_zenoh_connection(
            broker, self.topic, max_retries=3
        )

        sessions = []
        publishers = []
        pub_metadata = []
        for stream in amr_streams:
            broker = (
                f"tcp/{amr_streams[stream]['broker']}:{str(self.zenoh_port)}"
            )
            (session, pub, pub_met) = self.establish_zenoh_connection(
                broker, "zenoh-pub-stream", max_retries=3
            )
            sessions.append(session)
            publishers.append(pub)
            pub_metadata.append(pub_met)
            self.zenoh_port += 1

        # Standard GStreamer initialization
        Gst.init(None)

        print("Creating Pipeline \n ")
        pipeline = Gst.Pipeline()
        if not pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")

        print("Creating streammux \n ")
        # Create nvstreammux instance to form batches from one or more sources.
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")
        pipeline.add(streammux)

        for idx in range(number_sources):
            print("Creating source_bin ", idx, " \n ")
            uri_name = self.amr_stream_srcs[idx]
            if uri_name.find("rtsp://") == 0:
                is_live = True
            source_bin = self.create_source_bin(idx, uri_name)
            if not source_bin:
                sys.stderr.write("Unable to create source bin \n")
            pipeline.add(source_bin)
            padname = "sink_%u" % idx
            sinkpad = streammux.get_request_pad(padname)
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)

        if is_live:
            print("Atleast one of the sources is live")
            streammux.set_property("live-source", 1)

        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", number_sources)
        streammux.set_property("batched-push-timeout", 4000000)
        streammux.set_property("nvbuf-memory-type", mem_type)

        print("Creating Pgie \n ")
        nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not nvinfer:
            sys.stderr.write(" Unable to create pgie \n")
        nvinfer.set_property("config-file-path", "chemical_spill_config.txt")
        nvinfer_batch_size = nvinfer.get_property("batch-size")
        if nvinfer_batch_size != number_sources:
            print(
                "WARNING: Overriding infer-config batch-size "
                f"{nvinfer_batch_size} with number of sources "
                f"{number_sources} \n"
            )
            nvinfer.set_property("batch-size", number_sources)

        print("Creating nvvidconv \n ")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")
        nvvidconv.set_property("nvbuf-memory-type", mem_type)

        print("Creating nvosd \n ")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")
        # Note: display-mask is supported only for process-mode=0 (CPU)
        nvosd.set_property("display_mask", True)
        nvosd.set_property("process_mode", 0)

        print("Creating nvvidconv_1 \n ")
        nvvidconv_1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
        if not nvvidconv_1:
            sys.stderr.write(" Unable to create nvvidconv_1 \n")
        nvvidconv_1.set_property("nvbuf-memory-type", mem_type)

        print("Creating filter \n ")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        caps_filter = Gst.ElementFactory.make("capsfilter", "filter")
        if not caps_filter:
            sys.stderr.write(" Unable to get the caps filter \n")
        caps_filter.set_property("caps", caps)

        if self.deploy_single:
            # Make the encoder
            encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
            print("Creating H264 Encoder")
            if not encoder:
                sys.stderr.write(" Unable to create encoder")
            encoder.set_property("bitrate", 4000000)

            print("Creating H264 rtppay")
            rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
            if not rtppay:
                sys.stderr.write(" Unable to create rtppay")

            # Make the UDP sink
            print("Making UDP sink")
            sink = Gst.ElementFactory.make("udpsink", "udpsink")
            if not sink:
                sys.stderr.write(" Unable to create udpsink")
            sink.set_property("host", "0.0.0.0")
            sink.set_property("port", self.udp_port)
            sink.set_property("sync", False)
            sink.set_property("qos", 0)
        else:
            sink = Gst.ElementFactory.make("fakesink", "fakesink")
            sink.set_property("enable-last-sample", 0)
            sink.set_property("sync", 0)

        queue1 = Gst.ElementFactory.make("queue", "queue1")
        queue2 = Gst.ElementFactory.make("queue", "queue2")

        print("Adding elements to Pipeline \n")
        pipeline.add(queue1)
        pipeline.add(queue2)
        pipeline.add(nvinfer)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(caps_filter)
        pipeline.add(nvvidconv_1)
        if self.deploy_single:
            pipeline.add(encoder)
            pipeline.add(rtppay)
        pipeline.add(sink)

        print("Linking elements in the Pipeline \n")
        streammux.link(queue1)
        queue1.link(nvinfer)
        nvinfer.link(queue2)
        queue2.link(nvvidconv)
        nvvidconv.link(caps_filter)
        caps_filter.link(nvosd)
        nvosd.link(nvvidconv_1)
        if self.deploy_single:
            nvvidconv_1.link(encoder)
            encoder.link(rtppay)
            rtppay.link(sink)
        else:
            nvvidconv_1.link(sink)

        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        sink_pad_nvosd = nvvidconv_1.get_static_pad("sink")
        sink_pad_pgie = queue2.get_static_pad("sink")
        if not sink_pad_nvosd:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            sink_pad_nvosd.add_probe(
                Gst.PadProbeType.BUFFER,
                partial(self.overlay_pub_probe, publishers=publishers),
            )

        if not sink_pad_pgie:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            sink_pad_pgie.add_probe(
                Gst.PadProbeType.BUFFER, partial(self.spill_detection_probe)
            )

        # List the sources
        print("Now playing...")
        for i, source in enumerate(self.amr_stream_srcs):
            print(i, ": ", source)

        print("Starting pipeline \n")
        # start play back and listed to events
        pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass

        # cleanup
        print("Exiting app\n")
        pipeline.set_state(Gst.State.NULL)


def start_process(args):
    deploy_single, amr_stream_srcs, zenoh_port, udp_port, topic = args
    pipeline_main = DeepstreamSegmentation(
        deploy_single, amr_stream_srcs, zenoh_port, udp_port, topic
    )
    pipeline_main.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to input configuration yaml file",
        required=True,
    )

    args = parser.parse_args()
    stream_paths = args.config

    with open(stream_paths, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        amr_streams = config_data.get("robot_config", [])

    amr_stream_srcs = []
    topics = ""
    zenoh_port = 7447
    udp_port = 1234
    deploy_single = config_data.get("deploy_single")

    pool_size = len(amr_streams)
    with Pool(processes=pool_size) as pool:
        args_list = []
        udp_port = 1234

        for key in amr_streams:
            amr_stream_srcs.append(amr_streams[key]["url"])
            topics = amr_streams[key]["spill_topic"]
            args_list.append(
                (deploy_single, amr_stream_srcs, zenoh_port, udp_port, topics)
            )
            amr_stream_srcs = []
            topics = ""
            zenoh_port += 1
            udp_port += 1

        pool.map(start_process, args_list)
        pool.close()
        pool.join()
