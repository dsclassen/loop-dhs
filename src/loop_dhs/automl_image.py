# -*- coding: utf-8 -*-
import math
import os

import cv2


class AutoMLResult:
    """Just the values to be sent to DCSS"""

    def __init__(self, result: list) -> None:
        self.type = "LOOP_INFO"
        self.index = result[1]
        self.status = result[2]
        self.tipX = result[3]
        self.tipY = result[4]
        self.pinBaseX = result[5]
        self.fiberWidth = result[6]
        self.loopWidth = result[7]
        self.boxMinX = result[8]
        self.boxMaxX = result[9]
        self.boxMinY = result[10]
        self.boxMaxY = result[11]
        self.loopWidthX = result[12]
        self.isMicroMount = result[13]
        self.loopClass = result[14]
        self.loopScore = result[15]


class AutoMLImage:
    def __init__(self, message, output_dir) -> None:
        self.index = message.index
        self.axis_file_name = "loop_{:04}.jpeg".format(self.index)
        self.adorned_output_dir = os.path.join(output_dir, "bboxes")
        self.file_to_adorn = os.path.join(output_dir, self.axis_file_name)

        self.automl_score = message.loop_top_score
        self.automl_class = message.loop_top_classification

        if message.loop_num is not None:
            self.loop_upper_left = [message.loop_bb_minX, message.loop_bb_minY]
            self.loop_lower_right = [message.loop_bb_maxX, message.loop_bb_maxY]
            self.loop_tip = [message.tip_x, message.tip_y]
        else:
            self.loop_upper_left = [0.01, 0.01]
            self.loop_lower_right = [0.02, 0.02]
            self.loop_tip = [0.5, 0.5]

        if message.pin_num is not None:
            self.pin_upper_left = [message.pin_bb_minX, message.pin_bb_minY]
            self.pin_lower_right = [message.pin_bb_maxX, message.pin_bb_maxY]
        else:
            self.pin_upper_left = [0.03, 0.03]
            self.pin_lower_right = [0.04, 0.04]

    def adorn_image(self):
        self.draw_bounding_box()
        # self.draw_automl_stats()

    def draw_bounding_box(self):
        """Draw the AutoML bounding box and loop tip crosshair overlaid on a JPEG image."""
        image = cv2.imread(self.file_to_adorn)
        s = tuple(image.shape[1::-1])
        w = s[0]
        h = s[1]
        tipX_frac = self.loop_tip[0]
        tipY_frac = self.loop_tip[1]
        tipX = round(tipX_frac * w)
        tipY = round(tipY_frac * h)
        crosshair_size = round(0.1 * h)
        # upper left corner of rectangle in pixels.
        loop_start_point = (
            math.floor(self.loop_upper_left[0] * w),
            math.floor(self.loop_upper_left[1] * h),
        )
        pin_start_point = (
            math.floor(self.pin_upper_left[0] * w),
            math.floor(self.pin_upper_left[1] * h),
        )

        # lower right corner of rectangle in pixels.
        loop_end_point = (
            math.ceil(self.loop_lower_right[0] * w),
            math.ceil(self.loop_lower_right[1] * h),
        )
        pin_end_point = (
            math.ceil(self.pin_lower_right[0] * w),
            math.ceil(self.pin_lower_right[1] * h),
        )

        loop_w = round((self.loop_lower_right[0] - self.loop_upper_left[0]), 3)
        loop_h = round((self.loop_lower_right[1] - self.loop_upper_left[1]), 3)

        # volor in BGR
        red = (0, 0, 255)
        green = (0, 255, 0)
        magenta = (255, 0, 255)

        # Line thickness in px
        thickness = 1

        cv2.rectangle(image, loop_start_point, loop_end_point, red, thickness)
        cv2.rectangle(image, pin_start_point, pin_end_point, magenta, thickness)
        cross_hair_horz = [(tipX - crosshair_size, tipY), (tipX + crosshair_size, tipY)]
        cross_hair_vert = [(tipX, tipY - crosshair_size), (tipX, tipY + crosshair_size)]
        cv2.line(image, cross_hair_horz[0], cross_hair_horz[1], green, 2)
        cv2.line(image, cross_hair_vert[0], cross_hair_vert[1], green, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.70
        fontColor = red
        lineType = 2

        cv2.putText(
            image,
            self.automl_class,
            (20, 50),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        cv2.putText(
            image,
            ("TH: " + str(self.automl_score)),
            (20, 100),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        cv2.putText(
            image,
            ("tipX: " + str(tipX_frac)),
            (240, 50),
            font,
            fontScale,
            green,
            thickness,
            lineType,
        )
        cv2.putText(
            image,
            ("tipY: " + str(tipY_frac)),
            (240, 100),
            font,
            fontScale,
            green,
            thickness,
            lineType,
        )
        cv2.putText(
            image,
            ("loopW: " + str(loop_w)),
            (450, 50),
            font,
            fontScale,
            red,
            thickness,
            lineType,
        )
        cv2.putText(
            image,
            ("loopH: " + str(loop_h)),
            (450, 100),
            font,
            fontScale,
            red,
            thickness,
            lineType,
        )

        output_filename = "automl_" + os.path.basename(self.file_to_adorn)
        outfile = os.path.join(self.adorned_output_dir, output_filename)
        cv2.imwrite(outfile, image)
