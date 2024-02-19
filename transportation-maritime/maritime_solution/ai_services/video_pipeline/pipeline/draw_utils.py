# Created by Scalers AI for Dell Inc.

import cv2

class DrawUtils:
    def __init__(self):
        pass

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
        font_color = (255,255,255)
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
        rectangle_color = (179,179,186)
        border_thickness = -1
        cv2.rectangle(
            image, top_right, bottom_left, rectangle_color, border_thickness
        )

        # Calculate text positions
        top_text_position = (
            top_right[0] - max_text_width - text_padding,
            top_right[1] + text_padding + text_top_height,
        )

        bottom_text_x = (
            top_right[0]
            - max_text_width
            - (2 * text_padding)
            + (rectangle_width) // 2
            - (text_bottom_width // 2)
        )

        bottom_text_position = (
            bottom_text_x,
            top_right[1] + rectangle_height - text_padding,
        )

        # Put white text on the image
        font_color = (0,0,0)
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
