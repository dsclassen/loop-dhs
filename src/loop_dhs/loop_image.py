# -*- coding: utf-8 -*-
import csv
import math
import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, differential_evolution, minimize_scalar

from loop_dhs.automl_image import AutoMLResult


class LoopImageSet:
    """Class to hold set of JPEG images acquired via collectLoopImages operation.

    Attributes:
        images (list): List of images sent from AXIS video server
        results (list): List of results from AutoML

    """

    def __init__(self):
        self.images = []
        self.results = []
        self.automl_results = []
        self.header = [
            "LOOP_INFO",
            "index",
            "status",
            "tipX",
            "tipY",
            "pinBaseX",
            "fiberWidth",
            "loopWidth",
            "boxMinX",
            "boxMaxX",
            "boxMinY",
            "boxMaxY",
            "loopWidthX",
            "isMicroMount",
            "loopClass",
            "loopScore",
        ]
        self.data_frame = None
        self._number_of_images = None

    def add_image(self, image: bytes):
        """Add a jpeg image to the list of images."""
        self.images.append(image)
        self._number_of_images = len(self.images)

    def add_results(self, result: list):
        """
        Add the AutoML results to a list for use in reboxLoopImage
        loop_info stored as python list so this is a list of lists.
        """
        self.results.append(result)
        # self.automl_results.append(AutoMLResult(result))

    @property
    def number_of_images(self) -> int:
        """Get the number of images in the image list"""
        return self._number_of_images

    def write_csv_file(self, dir):
        fn = "loop_info.csv"
        results_file = os.path.join(dir, fn)
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")

            writer.writerow(self.header)
            for row in self.results:
                writer.writerow(row)

    def plot_loop_widths(self, dir):

        # def plot_loop_widths(results_dir: str, images: LoopImageSet):
        """
        Plot of image vs loopWidth.

        The curve fitting code is adapted from James Phillips.
        Here is a Python fitter with a sine equation and your data using the
        scipy.optimize Differential Evolution genetic algorithm module to determine
        initial parameter estimates for curve_fit's non-linear solver.
        https://stackoverflow.com/a/58478075/3023774
        """
        indices = [e[1] for e in self.results]
        # _logger.spam(f'PLOT INDICES: {indices}')
        _loop_widths = [e[7] for e in self.results]
        # _logger.spam(f'PLOT LOOP WIDTHS: {_loop_widths}')
        # empty list to store x y data
        _x_data = []
        _y_data = []

        # images are 2 degrees apart and must be converted to radians
        for index in indices:
            angle = index * 2
            rad = math.radians(angle)
            _x_data.append(rad)

        # convert loopWidth from fractional to pixel coordinates
        for width in _loop_widths:
            w = 480 * width
            _y_data.append(w)

        # convert python lists to numpy arrays
        x_data = np.array(_x_data)
        y_data = np.array(_y_data)

        # sine wave with amplitude, center, width, and offset
        def func(x, amplitude, center, width, offset):
            return amplitude * np.sin(np.pi * (x - center) / width) + offset

        # function for genetic algorithm to minimize (sum of squared error)
        def sum_of_squared_error(parameter_tuple):
            warnings.filterwarnings(
                "ignore"
            )  # do not print warnings by genetic algorithm
            val = func(x_data, *parameter_tuple)
            return np.sum((y_data - val) ** 2.0)

        # reasonable initial values are needed for a stable curve fit
        def generate_initial_parameters():
            # min and max used for bounds
            maxX = max(x_data)
            minX = min(x_data)
            maxY = max(y_data)
            minY = min(y_data)

            diffY = maxY - minY
            diffX = maxX - minX

            parameter_bounds = []
            parameter_bounds.append([0.0, diffY])  # search bounds for amplitude
            parameter_bounds.append([minX, maxX])  # search bounds for center
            parameter_bounds.append([0.0, diffX])  # search bounds for width
            parameter_bounds.append([minY, maxY])  # search bounds for offset

            # "seed" the np random number generator for repeatable results
            result = differential_evolution(
                sum_of_squared_error, parameter_bounds, seed=42
            )
            return result.x

        # by default, differential_evolution completes by calling curve_fit() using parameter bounds
        genetic_parameters = generate_initial_parameters()
        # _logger.debug(f'DIFFERENTIAL EVOLUTION: {genetic_parameters}')

        # now call curve_fit without passing bounds from the genetic algorithm,
        # just in case the best fit parameters are outside those bounds
        fitted_parameters, pcov = curve_fit(func, x_data, y_data, genetic_parameters)
        # _logger.debug(f'FITTED PARAMETERS: {fitted_parameters}')

        model_predictions = func(x_data, *fitted_parameters)

        abs_error = model_predictions - y_data

        SE = np.square(abs_error)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        r_squared = 1.0 - (np.var(abs_error) / np.var(y_data))

        # _logger.debug(f'RMSE: {RMSE}')
        # _logger.debug(f'R-SQUARED: {r_squared}')

        def model_and_scatter_plot(graph_width, graph_height):
            f = plt.figure(figsize=(graph_width / 100.0, graph_height / 100.0), dpi=100)
            axes = f.add_subplot(111)

            # raw data as a scatter plot
            axes.scatter(x_data, y_data, color="black", marker="o", label="data")

            # create data for the fitted equation plot
            x_model = np.linspace(min(x_data), max(x_data))
            y_model = func(x_model, *fitted_parameters)

            # now the model as a line plot
            axes.plot(x_model, y_model, color="red", label="fit")

            # calculate the phi value for the max (face view)
            fm = lambda xData: -func(xData, *fitted_parameters)
            res = minimize_scalar(fm, bounds=(0, 8))
            # _logger.info(f'LOOP MAX (FACE): {math.degrees(res.x)}')
            axes.plot(
                res.x,
                func(res.x, *fitted_parameters),
                color="green",
                marker="o",
                markersize=18,
            )
            max_x = str(round(math.degrees(res.x), 3))
            # max_y = str(round(-res.fun,3))
            face_label = "".join(["FACE: Phi = ", max_x, "\N{DEGREE SIGN}"])
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axes.annotate(face_label, (res.x + 0.2, -res.fun), fontsize=14, bbox=props)

            # calculate the phi value for the min (edge view)
            fm = lambda xData: func(xData, *fitted_parameters)
            res = minimize_scalar(fm, bounds=(0, 8))
            # _logger.info(f'LOOP MIN (EDGE): {math.degrees(res.x)}')
            axes.plot(
                res.x,
                func(res.x, *fitted_parameters),
                color="magenta",
                marker="o",
                markersize=18,
            )
            min_x = str(round(math.degrees(res.x), 2))
            # min_y = str(round(res.fun,2))
            edge_label = "".join(["EDGE: Phi = ", min_x, "\N{DEGREE SIGN}"])
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axes.annotate(edge_label, (res.x + 0.2, res.fun), fontsize=14, bbox=props)

            axes.set_xlabel("Image Phi Position (rad)")
            axes.set_ylabel("Loop Width (px)")
            axes.legend(loc="best")

            fn = "plot_loop_widths.png"
            results_plot = os.path.join(dir, fn)
            f.savefig(results_plot)

        graph_width = 800
        graph_height = 600
        model_and_scatter_plot(graph_width, graph_height)

    def plot_automl_stats(self, dir):
        indices = [e[1] for e in self.results]
        tipx = [e[3] for e in self.results]
        pinbasex = [e[5] for e in self.results]
        loopwidthx = [e[12] for e in self.results]
        scores = [e[15] for e in self.results]

        def scatter_plot(graph_width, graph_height):
            f = plt.figure(figsize=(graph_width / 100.0, graph_height / 100.0), dpi=100)
            axes = f.add_subplot(211)

            axes.scatter(indices, scores, marker="o", label="score")
            axes.scatter(indices, tipx, marker="o", label="tipx")
            axes.scatter(indices, pinbasex, marker="o", label="pinbasex")
            axes.scatter(indices, loopwidthx, marker="o", label="loopwidthx")

            axes.set_xlabel("Index")  # X axis data label
            axes.set_ylabel("")  # Y axis data label
            axes.legend(loc="best")

            axes2 = f.add_subplot(212)
            scores_array = np.asarray(scores)
            axes2.hist(scores_array)
            mu = scores_array.mean()
            median = np.median(scores_array)
            sigma = scores_array.std()
            textstr = "\n".join(
                (f"mu: {mu:4.3}", f"median: {median:4.3}", f"sigma: {sigma:4.3}")
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axes2.text(
                0.05,
                0.95,
                textstr,
                transform=axes2.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )

            fn = "plot_automl_scores.png"
            results_plot = os.path.join(dir, fn)
            f.savefig(results_plot)

        graph_width = 800
        graph_height = 1200
        scatter_plot(graph_width, graph_height)

    def pandas_plot(self, dir):
        self.data_frame = pd.DataFrame(self.results)
        self.data_frame.columns = self.header
        output_file_name = "pandas_plot"
        indices = self.data_frame["index"]
        scores = self.data_frame["loopScore"]
        tipx = self.data_frame["tipX"]
        pinbasex = self.data_frame["pinBaseX"]
        loopwidthx = self.data_frame["loopWidthX"]

        def plot(graph_width, graph_height):
            f = plt.figure(figsize=(graph_width / 100.0, graph_height / 100.0), dpi=100)
            axes = f.add_subplot(211)

            axes.scatter(indices, scores, marker="o", label="score")
            axes.scatter(indices, tipx, marker="o", label="tipx")
            axes.scatter(indices, pinbasex, marker="o", label="pinbasex")
            axes.scatter(indices, loopwidthx, marker="o", label="loopwidthx")

            axes.set_xlabel("Index")
            axes.set_ylabel("")
            axes.legend(loc="best")

            axes2 = f.add_subplot(212)
            self.data_frame["loopScore"].hist(ax=axes2, bins=20)
            mu = self.data_frame["loopScore"].mean()
            median = self.data_frame["loopScore"].median()
            sigma = self.data_frame["loopScore"].std()

            axes2.set_xlabel("AutoML confidence score")
            axes2.set_ylabel("count")

            textstr = "\n".join(
                (f"mean: {mu:4.4}", f"median: {median:4.4}", f"std: {sigma:4.4}")
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axes2.text(
                0.70,
                0.95,
                textstr,
                transform=axes2.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )

            fn = output_file_name + ".png"
            results_plot = os.path.join(dir, fn)
            f.savefig(results_plot)

        graph_width = 800
        graph_height = 1200
        plot(graph_width, graph_height)


class CollectLoopImageState:
    """Class to store state and objects for a single collectLoopImages operation."""

    def __init__(self):
        self._loop_images = LoopImageSet()
        self._image_index = 0
        self._automl_responses_received = 0
        self._results_dir = None
        self._done = False

    @property
    def loop_images(self):
        return self._loop_images

    @property
    def image_index(self) -> int:
        return self._image_index

    @image_index.setter
    def image_index(self, idx: int):
        self._image_index = idx

    @property
    def automl_responses_received(self) -> int:
        return self._automl_responses_received

    @automl_responses_received.setter
    def automl_responses_received(self, idx: int):
        self._automl_responses_received = idx

    @property
    def results_dir(self) -> str:
        return self._results_dir

    @results_dir.setter
    def results_dir(self, dir: str):
        self._results_dir = dir

    @property
    def done(self) -> bool:
        return self._done

    @done.setter
    def done(self, done_state: bool):
        self._done = done_state
