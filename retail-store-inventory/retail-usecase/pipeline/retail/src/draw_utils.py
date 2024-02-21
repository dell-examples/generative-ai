# Created by Scalers AI for Dell Inc.

import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class DrawUtils:
    def __init__(self, zones, objects, stream_name, colors, logger):
        """
        Initialize an instance of the class with configuration parameters.

        Args:
            zones (List): A list of zones or areas of interest for object tracking.
            objects (List): A list of objects or classes to be detected and tracked.
            stream_name (str): A unique identifier or name for the video stream.
            colors (Dict): A dictionary that maps object classes to their respective colors.
        """
        self.zones = zones
        self.objects = objects
        self.stream_name = stream_name
        self.colors = colors
        self.logger = logger

    def draw_rectangle_with_text(
        self,
        image,
        top_text,
        bottom_text,
        top_right,
        top_text_scale=0.8,
        bottom_text_scale=1.2,
        text_padding=20,
    ):
        """
        Draw a filled rectangle with two lines of text on an image.

        Args:
            image (numpy.ndarray): The image on which the rectangle and text will be drawn.
            top_text (str): The text to be displayed on the top line.
            bottom_text (str): The text to be displayed on the bottom line.
            top_right (tuple): The coordinates of the top-right corner of the rectangle.
            top_text_scale (float, optional): The scale factor for the top text size. Default is 0.8.
            bottom_text_scale (float, optional): The scale factor for the bottom text size. Default is 1.2.
            text_padding (int, optional): The padding between text and the rectangle edges. Default is 20.

        Returns:
            numpy.ndarray: The image with the rectangle and text added.
        """

        # Calculate the size of the text and the overall rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        # Calculate the dimensions of the top and bottom text
        (text_top_width, text_top_height), _ = cv2.getTextSize(
            top_text, font, top_text_scale, font_thickness
        )
        (text_bottom_width, text_bottom_height), _ = cv2.getTextSize(
            bottom_text, font, bottom_text_scale, font_thickness
        )

        # Calculate the size of the rectangle
        max_text_width = max(text_top_width, text_bottom_width)
        rectangle_width = max_text_width + 2 * text_padding
        rectangle_height = (
            text_padding * 3 + text_top_height + text_bottom_height
        )

        # Calculate the bottom left point of the rectangle
        bottom_left = (
            top_right[0] - rectangle_width,
            top_right[1] + rectangle_height,
        )

        # Draw the hollow rectangle
        cv2.rectangle(image, top_right, bottom_left, self.colors["black"], -1)

        # Calculate the horizontal center for the bottom text
        bottom_text_x = (
            top_right[0]
            - max_text_width
            - (2 * text_padding)
            + (rectangle_width) // 2
            - (text_bottom_width // 2)
        )
        bottom_text_y = top_right[1] + rectangle_height - text_padding

        # Calculate text positions
        top_text_position = (
            top_right[0] - max_text_width - text_padding,
            top_right[1] + text_padding + text_top_height,
        )
        bottom_text_position = (bottom_text_x, bottom_text_y)

        # Put white text on the image
        font_color = self.colors["white"]
        cv2.putText(
            image,
            top_text,
            top_text_position,
            font,
            top_text_scale,
            font_color,
            font_thickness,
        )
        cv2.putText(
            image,
            bottom_text,
            bottom_text_position,
            font,
            bottom_text_scale,
            font_color,
            font_thickness,
        )

        return image

    def draw_hollow_rectangle_with_text(
        self,
        image,
        top_text,
        bottom_text,
        top_right,
        top_text_scale=0.8,
        bottom_text_scale=1.2,
        text_padding=20,
    ):
        """
        Draw a hollow rectangle with two lines of text on an image.

        Args:
            image (numpy.ndarray): The image on which the rectangle and text will be drawn.
            top_text (str): The text to be displayed on the top line.
            bottom_text (str): The text to be displayed on the bottom line.
            top_right (tuple): The coordinates of the top-right corner of the rectangle.
            top_text_scale (float, optional): The scale factor for the top text size. Default is 0.8.
            bottom_text_scale (float, optional): The scale factor for the bottom text size. Default is 1.2.
            text_padding (int, optional): The padding between text and the rectangle edges. Default is 20.

        Returns:
            numpy.ndarray: The image with the hollow rectangle and text added.
        """

        # Calculate the size of the text and the overall rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        # Calculate the dimensions of the top and bottom text
        (text_top_width, text_top_height), _ = cv2.getTextSize(
            top_text, font, top_text_scale, font_thickness
        )
        (text_bottom_width, text_bottom_height), _ = cv2.getTextSize(
            bottom_text, font, bottom_text_scale, font_thickness
        )

        # Calculate the size of the rectangle
        max_text_width = max(text_top_width, text_bottom_width)
        rectangle_width = max_text_width + 2 * text_padding
        rectangle_height = (
            text_padding * 3 + text_top_height + text_bottom_height
        )

        # Calculate the bottom left point of the rectangle
        bottom_left = (
            top_right[0] - rectangle_width,
            top_right[1] + rectangle_height,
        )

        # Draw the hollow rectangle
        rectangle_color = self.colors["white"]
        border_thickness = 3
        cv2.rectangle(
            image, top_right, bottom_left, rectangle_color, border_thickness
        )

        # Calculate text positions
        top_text_position = (
            top_right[0] - max_text_width - text_padding,
            top_right[1] + text_padding + text_top_height,
        )
        bottom_text_position = (
            top_right[0] - max_text_width - text_padding,
            top_right[1] + rectangle_height - text_padding,
        )

        # Put white text on the image
        font_color = self.colors["white"]
        cv2.putText(
            image,
            top_text,
            top_text_position,
            font,
            top_text_scale,
            font_color,
            font_thickness,
        )
        cv2.putText(
            image,
            bottom_text,
            bottom_text_position,
            font,
            bottom_text_scale,
            font_color,
            font_thickness,
        )

        return image

    def draw_rectangles_with_text(
        self,
        image,
        main_text,
        top_texts,
        bottom_texts,
        coords,
        top_text_scale=0.6,
        bottom_text_scale=1.3,
        text_padding=20,
    ):
        """
        Draw multiple rectangles with text on an image.

        Args:
            image (numpy.ndarray): The image on which the rectangles and text will be drawn.
            main_text (str): The main text displayed at the center of the rectangles.
            top_texts (List[str]): A list of text to be displayed on top lines of the rectangles.
            bottom_texts (List[str]): A list of text to be displayed on the bottom lines of the rectangles.
            coords (tuple): The coordinates of the top-left corner of the first rectangle.
            top_text_scale (float, optional): The scale factor for the top text size. Default is 0.6.
            bottom_text_scale (float, optional): The scale factor for the bottom text size. Default is 1.3.
            text_padding (int, optional): The padding between text and the rectangle edges. Default is 20.

        Returns:
            numpy.ndarray: The image with the drawn rectangles and text.
        """

        # Calculate the size of the text and the overall rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        w_t = "".join(top_texts)
        w_b = "".join(bottom_texts)

        # Calculate the dimensions of the top and bottom text
        (text_top_width, text_top_height), _ = cv2.getTextSize(
            w_t, font, top_text_scale, font_thickness
        )
        (text_bottom_width, text_bottom_height), _ = cv2.getTextSize(
            w_b, font, bottom_text_scale, font_thickness
        )
        (main_width, main_height), _ = cv2.getTextSize(
            main_text, font, top_text_scale, font_thickness
        )

        # Calculate the size of the rectangle
        max_text_width = max(text_top_width, text_bottom_width)
        rectangle_width = (
            max_text_width + 4 * text_padding + 4 * font_thickness
        )
        rectangle_height = (
            text_padding * 4
            + text_top_height
            + text_bottom_height
            + main_height
            + 2 * font_thickness
        )

        # Calculate the bottom left point of the rectangle
        bottom_left = (
            coords[0] - rectangle_width,
            coords[1] + rectangle_height,
        )

        # Calculate the horizontal center for the main text
        main_text_x = coords[0] - (rectangle_width) // 2 - (main_width // 2)
        main_text_y = coords[1] + main_height + (text_padding // 2)

        cv2.rectangle(image, coords, bottom_left, self.colors["white"], -1)
        main_text_position = (main_text_x, main_text_y)
        cv2.putText(
            image,
            main_text,
            main_text_position,
            font,
            0.8,
            self.colors["black"],
            font_thickness,
        )

        bottom_text_width = cv2.getTextSize(
            top_texts[1], font, bottom_text_scale, font_thickness
        )[0][1]

        image = self.draw_rectangle_with_text(
            image,
            f"Current Count",
            bottom_texts[0],
            (
                coords[0]
                - 4 * bottom_text_width
                - (2 * text_padding)
                - (2 * font_thickness),
                coords[1] + main_height + text_padding + (2 * font_thickness),
            ),
            top_text_scale=0.6,
            bottom_text_scale=1.3,
            text_padding=20,
        )
        image = self.draw_rectangle_with_text(
            image,
            f"Total Count",
            bottom_texts[1],
            (
                coords[0],
                coords[1] + main_height + text_padding + (2 * font_thickness),
            ),
            top_text_scale=0.6,
            bottom_text_scale=1.3,
            text_padding=20,
        )

        return image

    def draw_shelf_bounding_boxes(self, frame, bounding_boxes):
        """
        Draw bounding boxes and associated information for shelf zones on an image.

        Args:
            frame (numpy.ndarray): The image on which bounding boxes and information will be drawn.
            bounding_boxes (List of Lists): A list of bounding boxes in the format [x1, y1, x2, y2].

        Returns:
            Tuple[numpy.ndarray, dict]: A tuple containing the modified image with drawn elements and a metadata dictionary.
        """

        idx = 0
        self.metadata = {}
        vals = {}
        x_coordinates = [1910, 1560]
        for zone_name, points_list in self.zones.items():
            points = np.array(
                [[point["x"], point["y"]] for point in points_list],
                dtype=np.int32,
            )
            color = self.colors["green"]
            cv2.polylines(
                frame, [points], isClosed=True, color=color, thickness=12
            )
            top_point = points[0]

            x, y = top_point

            text_label = zone_name
            font_label = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_label = 1
            font_thickness_label = 2
            text_size_label, _ = cv2.getTextSize(
                text_label, font_label, font_scale_label, font_thickness_label
            )
            rectangle_size_label = (
                text_size_label[0] + 20,
                text_size_label[1] + 20,
            )
            rectangle_top_left_label = (x, y)

            cv2.rectangle(
                frame,
                rectangle_top_left_label,
                (x + rectangle_size_label[0], y + text_size_label[1] + 20),
                self.colors["green"],
                -1,
            )
            cv2.putText(
                frame,
                text_label,
                (x + 10, y + text_size_label[1] + 10),
                font_label,
                font_scale_label,
                self.colors["white"],
                font_thickness_label,
            )

            polygon = Polygon(points)
            bbox_polygons = [
                Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
                for x1, y1, x2, y2 in bounding_boxes
            ]
            count_inside = sum(
                [polygon.contains(bbox) for bbox in bbox_polygons]
            )
            vals[zone_name] = {
                "total": self.objects[zone_name],
                "current": count_inside,
            }
            top_texts = [f"Current Count", f"Total Count"]
            bottom_texts = [str(count_inside), str(self.objects[zone_name])]
            frame = self.draw_rectangles_with_text(
                frame,
                zone_name,
                top_texts,
                bottom_texts,
                (x_coordinates[idx], 10),
                top_text_scale=0.6,
                bottom_text_scale=1.3,
                text_padding=20,
            )
            idx += 1
            for bbox in bounding_boxes:
                bbox_center = (
                    (bbox[0] + bbox[2]) // 2,
                    bbox[1],
                )  # Calculate the center of the top bounding line
                if polygon.contains(Point(bbox_center)):
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 255),
                        4,
                    )

        self.metadata[self.stream_name] = vals

        return frame, self.metadata
