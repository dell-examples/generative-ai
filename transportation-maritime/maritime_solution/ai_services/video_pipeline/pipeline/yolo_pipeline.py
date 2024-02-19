#!/usr/bin/env python3
# Created by Scalers AI for Dell Inc.

import sys
sys.path.append('./')
import gi
import logging

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
from common.bus_call import bus_call
from common.FPS import PERF_DATA
from functools import partial
import numpy as np
import pyds
import cv2
import os
import argparse
import zenoh
import json
import yaml
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from multiprocessing import Pool
from draw_utils import DrawUtils

class DeepstreamSegmentation:
    def __init__(self, rtsp_stream_srcs, broker_ip, zenoh_port, udp_port, streamid, zone_path):
        """
        Initialize the DeepstreamSegmentation instance.

        Args:
            rtsp_stream_srcs (list): List of RTSP stream sources.
            zenoh_port (int): Port for Zenoh communication.
            udp_port (int): Port for UDP communication.
            streamid (int): Stream ID.
            zone_path (str): Path to the zone.
        """
        self.rtsp_stream_srcs = rtsp_stream_srcs
        self.zone_path = zone_path
        self.streamid = streamid
        self.broker_ip = broker_ip
        self.zenoh_port = zenoh_port
        self.udp_port = udp_port
        self.perf_data = None
        self.GST_CAPS_FEATURES_NVMM = "memory:NVMM"
        self.stream_cnt = 0
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0
        self.total_fps = 0
        self.start_time = 0
        self.prev_time = time.time()
        self.coordinates_list = []
        self.polygon_list = []
        self.polygon_nplist = []
        print(self.broker_ip)
        # Initialize logger
        logging.basicConfig(
            filename="app.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger()
        self.draw_utils = DrawUtils()
        self.meta={"Total People":0, "Total Violations":0, "Stream ID":0} 
        # Initialize GStreamer
        Gst.init(None)
        self.logger.info(self.zenoh_port)
        self.logger.info(self.udp_port)
        

    def disable_overlay(self, obj_meta):
        """
        Disable text and bounding boxes overlays.

        Args:
            obj_meta: Object metadata containing overlay parameters.
        """
        obj_meta.rect_params.border_width = 0
        obj_meta.text_params.display_text = ""
        obj_meta.text_params.set_bg_clr = 0

    def is_onzone(self, obj_meta):
        """
        Check if the objects are inside the zones.

        Args:
            obj_meta: Object metadata containing information about the object.

        Returns:
            bool: True if the object is inside the zones, False otherwise.
        """
        on_zone = False
        zone_cond = False
        y_max = obj_meta.rect_params.top + obj_meta.rect_params.height
        point_x = (obj_meta.rect_params.width / 2) + obj_meta.rect_params.left
        person_point = Point((point_x, y_max))
        for index, poly in enumerate(self.polygon_list):
            if obj_meta.class_id == 0:
                zone_cond = poly.contains(person_point)
            if zone_cond:
                on_zone = True
        return on_zone

    def init_zones(self):
        """
        Initialize the restricted zone coordinated from the zone.json file

        The zone.json file should be updated with restricted zone coordinates
        before starting the application.
        """
        # Read the zone file
        with open(self.zone_path) as json_file:
            data = json.load(json_file)

        for j in range(len(data)):
            i = 0
            for i in range(len(data["zone" + str(j + 1)])):
                coordinates = [data["zone" + str(j + 1)][i]['x'], data["zone" + str(j + 1)][i]['y']]
                self.coordinates_list.append(coordinates)
            self.polygon_list.append(Polygon(self.coordinates_list))
            self.polygon_nplist.append(np.array([self.coordinates_list]))
            self.coordinates_list = []


    def draw_zones(self, buf_frame, color_index):
        """
        Overlay zone maps.

        Args:
            buf_frame (numpy.ndarray): Buffer frame to draw zones on.
            color_index (int): Index of the color in zone_colors to use for drawing.

        Returns:
            None
        """
        zone_colors = [
            (255, 0, 0),
            (0, 175, 0)
        ]
        for index, np_poly in enumerate(self.polygon_nplist):
            cv2.polylines(buf_frame, [np_poly], True, zone_colors[color_index], 5)
            _frame = buf_frame.copy()
            cv2.fillPoly(_frame, [np_poly], zone_colors[color_index])
            alpha = 0.4
            frame_width= 1920
            frame_height = 1080
            buf_frame[
                0:frame_height, 0:frame_width
            ] = cv2.addWeighted(_frame, alpha, buf_frame, 1 - alpha, 0)


    def establish_zenoh_connection(self, broker, key, max_retries=3):
        """
        Establishes a connection with Zenoh broker.

        Parameters:
        max_retries (int): Maximum number of connection retries (default: 3).

        Returns:
        tuple: Session, publisher, and metadata publisher objects upon successful connection.
        """
        retries = 0
        session = None
        pub = None
        pub_mets = None
        self.logger.info(broker)
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
                pub = session.declare_publisher(key)
                pub_mets = session.declare_publisher("metadata")
                self.logger.info("Zenoh broker connection established successfully.")
                return session, pub, pub_mets
            except Exception:
                if retries < max_retries - 1:
                    self.logger.error(f"Retrying to get the Zenoh broker connection ({retries + 1}/{max_retries})")
                retries += 1
                if retries < max_retries:
                    time.sleep(5)

        self.logger.error(f"Zenoh broker connection cannot be established after {max_retries} retries. Exiting.")
        if session is not None:
            session.close()
        exit(1)

    def sink_pad_buffer_probe(self, pad, info, u_data, pubs=None, mets=None):
        """
        Probe function to handle buffer events in the sink pad.

        Args:
            pad (Gst.Pad): The pad that received the buffer.
            info (Gst.PadProbeInfo): Information about the probe data.
            u_data (): User data.
            pubs (list): List of publishers to publish image data.
            mets (Queue): Queue to put metadata.

        Returns:
            Gst.PadProbeReturn: Return value indicating the action to take.
        """
        frame_number = 0
        num_rects = 0
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.logger.error("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                # Extract the data from the GStreamer buffer for the current frame
                frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num

            total = 0
            violations = 0
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
                if obj_meta.class_id == 0:
                    on_zone = self.is_onzone(obj_meta)
                    if on_zone and (obj_meta.confidence * 100) > 60:
                        total += 1
                        violations += 1
                    else:
                        total += 1

            frame = self.draw_utils.draw_hollow_rectangle_with_text(
                frame,
                f"Total People",
                f"{total}",
                (1910, 30),
                top_text_scale=0.6,
                bottom_text_scale=1.3,
                text_padding=20,
            )

            frame = self.draw_utils.draw_hollow_rectangle_with_text(
                frame,
                f"Current Violations",
                f"{violations}",
                (1750, 30),
                top_text_scale=0.6,
                bottom_text_scale=1.3,
                text_padding=20
            )                                                                                                            
            self.frame_count += 1

            # Calculate FPS every second
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 10.0:
                fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
                frame = self.draw_utils.draw_hollow_rectangle_with_text(
                    frame,
                    f"FPS: {fps:.2f}",
                    f"STREAM ID: {self.streamid}",
                    (400, 30),
                    top_text_scale=1,
                    bottom_text_scale=1,
                    text_padding=20,
                )
                self.prev_fps = fps
            else:
                frame = self.draw_utils.draw_hollow_rectangle_with_text(
                    frame,
                    f"FPS: {self.prev_fps:.2f}",
                    f"STREAM ID: {self.streamid}",
                    (400, 30),
                    top_text_scale=1,
                    bottom_text_scale=1,
                    text_padding=20,
                )

            image_buffer = frame.astype(np.uint8).tobytes()
            pubs[frame_meta.pad_index].put(image_buffer)
            mets.put(json.dumps(self.meta))

            stream_index = "stream{0}".format(frame_meta.pad_index)
            self.perf_data.update_fps(stream_index)
            self.stream_cnt += 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def sink_pad_buffer_probe_pgie(self, pad, info, u_data, pubs=None, mets=None):
        """
        Probe function to handle buffer events in the sink pad for PGIE.

        Args:
            pad (Gst.Pad): The pad that received the buffer.
            info (Gst.PadProbeInfo): Information about the probe data.
            u_data (): User data.
            pubs (list): List of publishers to publish image data.
            mets (Queue): Queue to put metadata.

        Returns:
            Gst.PadProbeReturn: Return value indicating the action to take.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.logger.error("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                # Extract the data from the GStreamer buffer for the current frame
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            persons_payload = []
            self.meta["Total People"] = 0
            self.meta["Total Violations"] = 0
            self.meta["Stream ID"] = self.streamid
            person_count = 0
            zone_state = False
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
                if obj_meta.class_id == 0:
                    on_zone = self.is_onzone(obj_meta)
                    if on_zone and (obj_meta.confidence * 100) > 60:
                        zone_state = True
                        if on_zone:
                            obj_meta.rect_params.border_color.set(
                                1.0, 0.0, 0.0, 10
                            )
                            person_count += 1
                            self.meta["Total Violations"] += 1
                            self.meta["Total People"] += 1
                        else:
                            obj_meta.rect_params.border_color.set(
                                0.0, 1.0, 0.0, 10
                            )
                            self.meta["Total People"] += 1
                        persons_payload.append(
                            {
                                "on_zone": on_zone,
                                "class": obj_meta.class_id,
                                "conf": obj_meta.confidence,
                                "bbox": {
                                    "top": obj_meta.rect_params.top,
                                    "left": obj_meta.rect_params.left,
                                    "width": obj_meta.rect_params.width,
                                    "height": obj_meta.rect_params.height
                                }
                            }
                        )
                    else:
                        obj_meta.rect_params.border_color.set(
                            0.0, 1.0, 0.0, 10
                        )
                else:
                    self.disable_overlay(obj_meta)
            buf_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            if zone_state:
                self.draw_zones(buf_frame,0)
            else:
                self.draw_zones(buf_frame,1)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK



    def cb_newpad(self, decodebin, decoder_src_pad, data):
        """Callback function for handling new pads added to decodebin.

        Args:
            decodebin (Gst.Element): The decodebin element.
            decoder_src_pad (Gst.Pad): The source pad of the decoder.
            data (): User data.

        Returns:
            None
        """
        self.logger.info("In cb_newpad\n")
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        if gstname.find("video") != -1:
            if features.contains("memory:NVMM"):
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
            else:
                sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        """
        Callback function to handle child-added signal from decodebin.

        Args:
            child_proxy (GObject.Object): The child object.
            Object (Gst.Element): The decodebin element.
            name (str): Name of the child.
            user_data (): User data.

        Returns:
            None
        """
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)

        if "source" in name:
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property('drop-on-latency') is not None:
                Object.set_property("drop-on-latency", True)

    def create_source_bin(self, index, uri):
        """Create a source bin with a uridecodebin element.

        Args:
            index (int): Index of the source bin.
            uri (str): URI of the media source.

        Returns:
            Gst.Bin: The created source bin.
        """

        self.logger.info("Creating source bin")

        bin_name = "source-bin-%02d" % index
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)

        nbin.add(uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin

    def run(self):
        """
        Run the DeepstreamSegmentation application.

        Initializes zones, establishes communication channels, and sets up the GStreamer pipeline for real-time video processing.
        The pipeline includes components for video source handling, inference, object detection, overlay rendering, encoding, and streaming.
        """
        self.init_zones()

            
        self.perf_data = PERF_DATA(len(self.rtsp_stream_srcs))
        number_sources = len(self.rtsp_stream_srcs)
        sessions= []
        pubs = []
        pub_mets = []
        zenoh_port = self.zenoh_port
            
        key = f"zenoh-pub-stream"
        # IPAddr = os.getenv("SERVER_IP", "localhost")
        broker = f"tcp/{self.broker_ip}:{str(zenoh_port)}"
        
        (session, pub, pub_met) = self.establish_zenoh_connection(broker, key, max_retries=3)
        sessions.append(session)
        pubs.append(pub)
        pub_mets.append(pub_met)
            
        # Standard GStreamer initialization
        Gst.init(None)

        # Create gstreamer elements */
        # Create Pipeline element that will form a connection of other elements
        self.logger.info("Creating Pipeline \n ")
        pipeline = Gst.Pipeline()
        is_live = False

        if not pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")
        self.logger.info("Creating streammux \n ")

        # Create nvstreammux instance to form batches from one or more sources.
        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")

        pipeline.add(streammux)
        for i in range(number_sources):
            uri_name = self.rtsp_stream_srcs[i]
            if uri_name.find("rtsp://") == 0:
                is_live = True
            source_bin = self.create_source_bin(i, uri_name)
            if not source_bin:
                sys.stderr.write("Unable to create source bin \n")
            pipeline.add(source_bin)
            padname = "sink_%u" % i
            sinkpad = streammux.get_request_pad(padname)
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)

        self.logger.info("Creating Pgie \n ")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie \n")

        self.logger.info("Creating nvvidconv \n ")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")

        self.logger.info("Creating nvosd \n ")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")

        if is_live:
            self.logger.info("Atleast one of the sources is live")
            streammux.set_property('live-source', 1)

        self.logger.info("Creating nvvidconv1 \n ")
        nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
        if not nvvidconv1:
            sys.stderr.write(" Unable to create nvvidconv1 \n")

        self.logger.info("Creating filter1 \n ")
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
        if not filter1:
            sys.stderr.write(" Unable to get the caps filter1 \n")
        filter1.set_property("caps", caps1)

        self.logger.info("Creating filter2 \n ")
        caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter2 = Gst.ElementFactory.make("capsfilter", "filter2")
        if not filter2:
            sys.stderr.write(" Unable to get the caps filter2 \n")
        filter2.set_property("caps", caps2)
        
        # Make the encoder
        encoder = Gst.ElementFactory.make("x264enc", "encoder1")
        self.logger.info("Creating H264 Encoder")
        if not encoder:
            sys.stderr.write(" Unable to create encoder")
        encoder.set_property("bitrate", 4000000)

        # Make the payload-encode video into RTP packets
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        self.logger.info("Creating H264 rtppay")
        if not rtppay:
            sys.stderr.write(" Unable to create rtppay")
        
        
        # Make the UDP sink
        sink = Gst.ElementFactory.make("udpsink", "udpsink")
        if not sink:
           sys.stderr.write(" Unable to create udpsink")
        IPAddr = os.getenv("VISUALIZATION_SERVER_IP", "0.0.0.0")
        sink.set_property("host", IPAddr)
        sink.set_property("port", self.udp_port)
        sink.set_property("sync", False)
        sink.set_property("qos", 0)            

        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', number_sources)
        streammux.set_property('batched-push-timeout', 4000000)
        pgie.set_property('config-file-path', "dstest_segmask_config.txt")
        pgie_batch_size = pgie.get_property("batch-size")
        if (pgie_batch_size != number_sources):
            self.logger.info("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
                number_sources, " \n")
            pgie.set_property("batch-size", number_sources)

        nvosd.set_property("display_bbox", True) # Note: display-mask is supported only for process-mode=0 (CPU)
        nvosd.set_property('process_mode', 1)


        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)

        queue1=Gst.ElementFactory.make("queue","queue1")
        queue2=Gst.ElementFactory.make("queue","queue2")

        pipeline.add(queue1)
        pipeline.add(queue2)
        self.logger.info("Adding elements to Pipeline \n")
        pipeline.add(pgie)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(filter1)
        pipeline.add(filter2)
        pipeline.add(nvvidconv1)
        pipeline.add(encoder)
        pipeline.add(rtppay)
        pipeline.add(sink)

        self.logger.info("Linking elements in the Pipeline \n")
        streammux.link(pgie)
        pgie.link(queue2)
        queue2.link(nvvidconv)
        nvvidconv.link(filter1)
        filter1.link(queue1)
        queue1.link(nvosd)
        nvosd.link(nvvidconv1)
        nvvidconv1.link(encoder)
        encoder.link(rtppay)
        rtppay.link(sink)
        
        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        sink_pad_nvosd = queue1.get_static_pad("sink")
        sink_pad_pgie = filter1.get_static_pad("sink")
        if not sink_pad_nvosd:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            sink_pad_nvosd.add_probe(Gst.PadProbeType.BUFFER, partial(self.sink_pad_buffer_probe, u_data=0, pubs=pubs, mets=pub_met))
            GLib.timeout_add(3000, self.perf_data.perf_print_callback)

        if not sink_pad_pgie:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            sink_pad_pgie.add_probe(Gst.PadProbeType.BUFFER, partial(self.sink_pad_buffer_probe_pgie, u_data=0))
            
        # List the sources
        self.logger.info("Now playing...")

        self.logger.info("Starting pipeline \n")
        # start play back and listed to events		
        pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass
        # cleanup
        self.logger.info("Exiting app\n")
        pipeline.set_state(Gst.State.NULL)

def start_process(args):
    """
    Start the DeepstreamSegmentation application.

    Args:
        args (tuple): A tuple containing the following elements:
            - rtsp_stream_srcs (list): List of RTSP stream sources.
            - zenoh_port (int): Port number for Zenoh communication.
            - udp_port (int): Port number for UDP communication.
            - streamid (int): ID of the stream.
            - zone_path (str): Path to the zone configuration file.
    """
    rtsp_stream_srcs, broker_ip, zenoh_port, udp_port , streamid, zone_path = args
    pipeline_main = DeepstreamSegmentation(rtsp_stream_srcs, broker_ip, zenoh_port, udp_port, streamid, zone_path)
    pipeline_main.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="deepstream_segmask.py", 
                description="deepstream-segmask takes multiple URI streams as input" \
                    " and re-sizes and binarizes segmentation mask arrays to save to image")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input yaml",
        metavar="URIs",
        required=True,
    )

    parser.add_argument(
        "-z",
        "--zenoh_start",
        help="Zenoh starting port",
        metavar="Zenoh starting port",
        required=True,
    )

    parser.add_argument(
        "-u",
        "--udp_start",
        help="UDP starting port",
        metavar="UDP starting port",
        required=True,
    )

    args = parser.parse_args()
    stream_paths = args.input
    
    with open(stream_paths, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        rtsp_streams = config_data.get("rtsp_streams", [])

    rtsp_stream_srcs = []
    zenoh_port = int(args.zenoh_start)
    udp_port = int(args.udp_start)
    
    pool_size = len(rtsp_streams)
    with Pool(processes=pool_size) as pool:
        args_list = []
        udp_port = int(args.udp_start)
        zenoh_port = int(args.zenoh_start)
        streamid = 1
        for key in rtsp_streams:
            rtsp_stream_srcs.append(rtsp_streams[key]["url"])
            args_list.append((rtsp_stream_srcs, rtsp_streams[key]["broker"], zenoh_port, udp_port, f"UID1{streamid}", rtsp_streams[key]["zone"]))
            rtsp_stream_srcs = []
            zenoh_port += 1
            udp_port += 1
            streamid += 1
        pool.map(start_process, args_list)
        pool.close()
        pool.join()

