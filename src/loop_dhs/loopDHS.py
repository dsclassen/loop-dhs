# -*- coding: utf-8 -*-
"""
loopDHS
"""
__author__ = "Scott Classen"
__copyright__ = "Scott Classen"
__license__ = "mit"

import glob
import io
import logging
import os
import re
import signal
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import coloredlogs
import verboselogs
import yaml
from pydhsfw.automl import AutoMLPredictRequest, AutoMLPredictResponse
from pydhsfw.axis import AxisImageRequestMessage, AxisImageResponseMessage
from pydhsfw.dcss import (
    DcssContext,
    DcssHtoSClientIsHardware,
    DcssHtoSOperationCompleted,
    DcssHtoSOperationUpdate,
    DcssStoCSendClientType,
    DcssStoHRegisterOperation,
    DcssStoHStartOperation,
    register_dcss_start_operation_handler,
)
from pydhsfw.dhs import Dhs, DhsContext, DhsInit, DhsStart
from pydhsfw.jpeg_receiver import JpegReceiverImagePostRequestMessage
from pydhsfw.processors import Context, register_message_handler

from loop_dhs import __version__
from loop_dhs.automl_image import AutoMLImage
from loop_dhs.loop_dhs_config import LoopDHSConfig, LoopDHSState
from loop_dhs.loop_image import CollectLoopImageState

_logger = verboselogs.VerboseLogger("loopDHS")


@register_message_handler("dhs_init")
def dhs_init(message: DhsInit, context: DhsContext):
    """DHS initialization handler function."""
    parser = message.parser

    parser.add_argument(
        "--version",
        action="version",
        version="loopDHS version {ver}".format(ver=__version__),
    )
    parser.add_argument(
        dest="beamline",
        help="Beamline Name (e.g. BL-831 or SIM831). This determines which beamline-specific \
              config file to load from config directory.",
        metavar="Beamline",
    )
    parser.add_argument(
        dest="dhs_name",
        help="Optional alternate DHS Name (e.g. what dcss is expecting this DHS to be named). \
              If omitted then this value is set to be the name of this script.",
        metavar="DHS Name",
        nargs="?",
        default=Path(__file__).stem,
    )
    parser.add_argument(
        "-v",
        dest="verbosity",
        help="Sets the chattiness of logging (none to -vvvv)",
        action="count",
        default=0,
    )

    args = parser.parse_args(message.args)

    logfile = configure_logging(args.verbosity)

    conf_file = "config/" + args.beamline + ".config"
    with open(conf_file, "r") as f:
        yconf = yaml.safe_load(f)
        context.config = LoopDHSConfig(yconf)
    context.config["DHS"] = args.dhs_name

    loglevel_name = logging.getLevelName(_logger.getEffectiveLevel())

    context.state = LoopDHSState()

    if context.config.save_images:
        context.config.make_debug_dir()

    _logger.success("=============================================")
    _logger.success("Initializing DHS")
    _logger.success(f"Start Time:         {datetime.now()}")
    _logger.success(f"Logging level:      {loglevel_name}")
    _logger.success(f"Log File:           {logfile}")
    _logger.success(f"Config file:        {conf_file}")
    _logger.success(f'Initializing:       {context.config["DHS"]}')
    _logger.success(f'DCSS HOST:          {context.config["dcss.host"]}')
    _logger.success(f'     PORT:          {context.config["dcss.port"]}')
    _logger.success(f'AUTOML HOST:        {context.config["loopdhs.automl.host"]}')
    _logger.success(f'       PORT:        {context.config["loopdhs.automl.port"]}')
    _logger.success(
        f'JPEG RECEIVER PORT: {context.config["loopdhs.jpeg_receiver.port"]}'
    )
    _logger.success(f'AXIS HOST:          {context.config["loopdhs.axis.host"]}')
    _logger.success(f'     PORT:          {context.config["loopdhs.axis.port"]}')
    _logger.success("=============================================")


@register_message_handler("dhs_start")
def dhs_start(message: DhsStart, context: DhsContext):
    """DHS start handler"""
    # Connect to DCSS
    context.create_connection("dcss_conn", "dcss", context.config.dcss_url)
    context.get_connection("dcss_conn").connect()

    # Connect to GCP AutoML docker service
    context.create_connection(
        "automl_conn",
        "automl",
        context.config.automl_url,
        {"heartbeat_path": "/v1/models/default"},
    )
    context.get_connection("automl_conn").connect()

    # Connect to an AXIS Video Server
    context.create_connection("axis_conn", "axis", context.config.axis_url)
    context.get_connection("axis_conn").connect()

    # Create a jpeg receiving port. Only connect when ready to receive images.
    context.create_connection(
        "jpeg_receiver_conn", "jpeg_receiver", context.config.jpeg_receiver_url
    )
    # context.get_connection('jpeg_receiver_conn').connect()


@register_message_handler("stoc_send_client_type")
def dcss_send_client_type(message: DcssStoCSendClientType, context: Context):
    """Send client type to DCSS during initial handshake."""
    context.get_connection("dcss_conn").send(
        DcssHtoSClientIsHardware(context.config["DHS"])
    )


@register_message_handler("stoh_register_operation")
def dcss_reg_operation(message: DcssStoHRegisterOperation, context: Context):
    """Register the operations that DCSS has assigned to thsi DHS."""
    # Need to deal with unimplemented operations
    _logger.success(f"REGISTER: {message}")


@register_message_handler("stoh_start_operation")
def dcss_start_operation(message: DcssStoHStartOperation, context: Context):
    """Handle incoming requests to start an operation."""
    _logger.info(f"FROM DCSS: {message}")
    op = message.operation_name
    opid = message.operation_handle
    _logger.debug(f"OPERATION: {op}, HANDLE: {opid}")


@register_dcss_start_operation_handler("testAutoML")
def predict_one(message: DcssStoHStartOperation, context: DcssContext):
    """
    The operation is for testing AutoML. It reads a single image of a nylon
    loop from the tests directory and sends it to AutoML.
    """
    activeOps = context.get_active_operations(message.operation_name)
    _logger.debug(f"Active operations pre-completed={activeOps}")
    context.get_connection("dcss_conn").send(
        DcssHtoSOperationUpdate(
            message.operation_name,
            message.operation_handle,
            "about to predict one test image",
        )
    )
    image_key = "THE-TEST-IMAGE"
    filename = "tests/loop_nylon.jpg"
    with io.open(filename, "rb") as image_file:
        binary_image = image_file.read()
    context.get_connection("automl_conn").send(
        AutoMLPredictRequest(image_key, binary_image)
    )


@register_dcss_start_operation_handler("collectLoopImages")
def collect_loop_images(message: DcssStoHStartOperation, context: DcssContext):
    """
    Collects a set of JPEG images for analysis by AutoML.
    Initiates the the jpeg receiver, informs DCSS to start_oscillation, and
    as each jpeg image file is received it is processed and the AutoML results
    sent back as operation updates.

    DCSS may send a single arg <pinBaseSizeHint>, but I think we can ignore it.
    """
    # 1. Instantiate CollectLoopImageState.
    context.get_active_operations(
        operation_name=message.operation_name, operation_handle=message.operation_handle
    )[0].state = CollectLoopImageState()
    opName = message.operation_name
    opHandle = message.operation_handle

    # 2. Set image collection flag.
    context.state.collect_images = True

    # 3. Open the JPEG receiver port.
    context.get_connection("jpeg_receiver_conn").connect()

    # make a RESULTS directory for this instance of the operation.
    if context.config.save_images:
        if os.path.exists(context.config.timestamped_debug_dir):
            opDir = "".join([opName, opHandle.replace(".", "_")])
            operationResultsDir = os.path.join(
                context.config.timestamped_debug_dir, opDir
            )
            boundingBoxDir = os.path.join(operationResultsDir, "bboxes")

            activeOp = context.get_active_operations(operation_name="collectLoopImages")
            activeOp[0].state.results_dir = operationResultsDir

            os.makedirs(operationResultsDir)
            os.makedirs(boundingBoxDir)
            _logger.debug(f"SAVING RAW JPEG IMAGES TO: {operationResultsDir}")
            _logger.debug(f"SAVING OPENCV ADORNED IMAGES TO: {operationResultsDir}")
        else:
            _logger.error("RESULTS FOLDER MISSING")

    # 4. Send operation update message to DCSS to trigger sample rotation and axis server.
    context.get_connection("dcss_conn").send(
        DcssHtoSOperationUpdate(
            message.operation_name, message.operation_handle, "start_oscillation"
        )
    )


@register_dcss_start_operation_handler("getLoopTip")
def get_loop_tip(message: DcssStoHStartOperation, context: DcssContext):
    """
    Returns the position of the right-most point of a loop.
    Takes a single optional integer arg <ifaskPinPosFlag>

    1. htos_operation_completed getLoopTip operation_handle normal tipX tipY
       (when ifaskPinPosFlag = 0)
       or:
       htos_operation_completed getLoopTip operation_handle normal tipX tipY pinBaseX
       (when ifaskPinPosFlag = 1)

    2. htos_operation_completed getLoopTip operation_handle error TipNotInView +/-
    """
    # need to confirm that pinBaseX is the same as PinPos in imgCentering.cc
    #
    # 1. Request single jpeg image from axis video server. takes camera as arg.
    cam = str(context.config.axis_camera)
    context.get_connection("axis_conn").send(
        AxisImageRequestMessage("".join(["camera=", cam]))
    )


@register_dcss_start_operation_handler("getLoopInfo")
def get_loop_info(message: DcssStoHStartOperation, context: DcssContext):
    """
    This operation should return full suite of info about a single image.

    DCSS may send a single arg pinBaseSizeHint, but I think we can ignore it.
    """
    cam = str(context.config.axis_camera)
    context.get_connection("axis_conn").send(
        AxisImageRequestMessage("".join(["camera=", cam]))
    )


@register_dcss_start_operation_handler("stopCollectLoopImages")
def stop_collect_loop_images(message: DcssStoHStartOperation, context: DcssContext):
    """
    Sets a global flag to signal collectLoopImages to stop and optionally to
    shutdown the jpeg receiver.
    """
    # 1. Set image collection to False.
    context.state.collect_images = False

    # 2. Shutdown JPEG receiver port.
    # try moving this to automl_predict_response handler
    # context.get_connection('jpeg_receiver_conn').disconnect()

    # 3. Send operation completed message to DCSS
    context.get_connection("dcss_conn").send(
        DcssHtoSOperationCompleted(
            message.operation_name, message.operation_handle, "normal", "flag set"
        )
    )


@register_dcss_start_operation_handler("reboxLoopImage")
def rebox_loop_image(message: DcssStoHStartOperation, context: DcssContext):
    """
    Refine the loop bounding box. I'm not sure of it's use with AutoML loop
    prediction, but it was important for the original edge detection AI developed at SSRL.

    Parameters:
    index (int): which image we want to inspect
    start (double): X position start. Used for bracket. We will not use this.
    end (double): X position end. Used for bracket. We will not use this.

    e.g. reboxLoopImage 1.4 43 0.517685 0.563561

    Returns:
    returnIndex
    resultMinY
    resultMaxY
    (resultMaxY - resultMinY) <--- loopWidth
    """
    rebox_image = int(message.operation_args[0])
    previous_results = context.state.rebox_images.results[rebox_image]
    _logger.info(f"REQUEST REBOX OF IMAGE: {rebox_image} RESULTS: {previous_results}")
    index = previous_results[1]
    loopWidth = previous_results[7]
    boxMinY = previous_results[10]
    boxMaxY = previous_results[11]
    results = [index, boxMinY, boxMaxY, loopWidth]
    # transmorgrify into space-seperated list for Tcl.
    return_msg = " ".join(map(str, results))
    context.get_connection("dcss_conn").send(
        DcssHtoSOperationCompleted(
            message.operation_name, message.operation_handle, "normal", return_msg
        )
    )


@register_message_handler("automl_predict_response")
def automl_predict_response(message: AutoMLPredictResponse, context: DcssContext):
    """
    Process inference results returned from AutoML.
    """

    activeOps = context.get_active_operations()
    _logger.debug(f"Active operations pre-completed={activeOps}")

    # AutoML results filtering.
    # fail immediatly if score for top object is below threshold.
    # I think this happens if jpeg is completely blank.
    if message.get_score(0) < context.config.automl_thhreshold:
        _logger.warning(
            f"AUTOML SCORE: {message.get_score(0)} "
            f"BELOW THRESHOLD: {context.config.automl_thhreshold}"
        )
        status = "failed"
        result = ["no loop or pin detected, AutoML score: ", message.get_score(0)]

    else:
        # look at the top N results for a pin and a loop object.
        # This code should find the highest scoring pin and loop.
        for i in range(context.config.automl_scan_n_results):
            object = message.get_detection_class_as_text(i)
            score = message.get_score(i)

            if object == "pin" and message.pin_num is None:
                message.pin_num = i
                _logger.info(f"AUTOML RESULT #{i} IS A: {object: <8} SCORE: {score}")

            elif (
                object == "mitegen" or object == "nylon"
            ) and message.loop_num is None:
                message.loop_num = i
                if score < context.config.automl_thhreshold:
                    _logger.info(
                        f"AUTOML RESULT #{i} IS A: {object: <8} SCORE: {score} "
                        f"BELOW TH: {context.config.automl_thhreshold}"
                    )
                else:
                    _logger.info(
                        f"AUTOML RESULT #{i} IS A: {object: <8} SCORE: {score}"
                    )
            _logger.spam(f"{i} {object=} {score=}")

        # if no loop found in top N results
        if message.loop_num is None:
            _logger.warning(
                f"NO LOOP IN TOP {context.config.automl_scan_n_results} AUTOML RESULTS. SETTING TO 0"
            )
            message.loop_num = 0

        # if message.pin_num is None:
        #     _logger.warning('NO PIN IN TOP 5 AUTOML RESULTS. SETTING TO 0')
        #     message.pin_num = 0

    for ao in activeOps:
        if ao.operation_name == "testAutoML":
            result = [
                message.image_key,
                message.loop_top_score,
                message.loop_top_bb,
                message.loop_top_classification,
                message.loop_top_score,
            ]
            msg = " ".join(map(str, result))
            _logger.info(f"SEND TO DCSS: {msg}")
            context.get_connection("dcss_conn").send(
                DcssHtoSOperationCompleted(
                    ao.operation_name, ao.operation_handle, status, msg
                )
            )

        elif ao.operation_name == "getLoopTip":
            result = [message.tip_x, message.tip_y]
            msg = " ".join(map(str, result))
            _logger.info(f"SEND TO DCSS: {msg}")
            context.get_connection("dcss_conn").send(
                DcssHtoSOperationCompleted(
                    ao.operation_name, ao.operation_handle, message.status, msg
                )
            )

        elif ao.operation_name == "getLoopInfo":

            msg = " ".join(map(str, message.loop_info_result))
            _logger.info(f"SEND TO DCSS: {msg}")
            context.get_connection("dcss_conn").send(
                DcssHtoSOperationCompleted(
                    ao.operation_name, ao.operation_handle, "normal", msg
                )
            )

        elif ao.operation_name == "collectLoopImages":

            # Increment AutoML responses received.
            ao.state.automl_responses_received += 1
            received = ao.state.automl_responses_received
            sent = ao.state.image_index
            expected_frames = context.config.osci_time * context.config.video_fps
            collect = context.state.collect_images

            # Send Operation Update message.
            # if received < expected_frames and collect is True:
            if received < expected_frames:
                _logger.debug(
                    f"OPERATION UPDATE SENT TO AutoML: {sent} "
                    f"RECEIVED FROM AutoML: {received} "
                    f"COLLECT: {collect} INDEX: {message.index}"
                )

                msg = " ".join(map(str, message.dcss_result))
                ao.state.loop_images.add_results(message.dcss_result)
                _logger.success(f"OPERATION UPDATE SEND TO DCSS1: {msg}")
                context.get_connection("dcss_conn").send(
                    DcssHtoSOperationUpdate(ao.operation_name, ao.operation_handle, msg)
                )

                # Draw the AutoML bounding box if we are saving files to disk.
                if context.config.save_images:
                    image = AutoMLImage(message, ao.state.results_dir)
                    image.adorn_image()
            # Send Operation Complete message.
            elif received >= expected_frames:
                _logger.success(f"SENT: {sent} RECEIVED: {received} COLLECT: {collect}")
                if context.config.save_images:
                    ao.state.loop_images.write_csv_file(ao.state.results_dir)
                    ao.state.loop_images.plot_loop_widths(ao.state.results_dir)
                    ao.state.loop_images.pandas_plot(ao.state.results_dir)
                context.state.rebox_images = ao.state.loop_images
                _logger.info("SEND OPERATION COMPLETE TO DCSS")
                context.get_connection("dcss_conn").send(
                    DcssHtoSOperationCompleted(
                        ao.operation_name, ao.operation_handle, "normal", "done"
                    )
                )
            # Not sure we can ever get to this bit of code
            else:
                _logger.warning("=====================================================")
                _logger.warning(f"SENT: {sent} RECEIVED: {received} COLLECT: {collect}")
                _logger.warning("=====================================================")

    activeOps = context.get_active_operations()
    _logger.debug(f"Active operations post-completed={activeOps}")


@register_message_handler("jpeg_receiver_image_post_request")
def jpeg_receiver_image_post_request(
    message: JpegReceiverImagePostRequestMessage, context: DhsContext
):
    """Handles JPEG images arriving on the jpeg receiver port
    then sends them to AutoMLPredictRequest."""
    # _logger.spam(message.file)
    activeOps = context.get_active_operations(operation_name="collectLoopImages")

    if len(activeOps) > 0:
        activeOp = activeOps[0]
        opName = activeOp.operation_name
        opHandle = activeOp.operation_handle
        resultsDir = activeOp.state.results_dir

        _logger.debug(f"ADD {len(message.file)} BYTE IMAGE TO JPEG LIST")
        activeOp.state.loop_images.add_image(message.file)

        if context.config.save_images:
            save_jpeg(message.file, activeOp.state.image_index, resultsDir)

        image_key = ":".join([opName, opHandle, str(activeOp.state.image_index)])

        context.get_connection("automl_conn").send(
            AutoMLPredictRequest(image_key, message.file)
        )
        _logger.debug(f"IMAGE_KEY: {image_key}")
        # increment image index which we use as a count of images SENT.
        activeOp.state.image_index += 1
    else:
        _logger.warning(
            "RECEVIED JPEG, BUT NOT DOING ANYTHING WITH IT. "
            "no active collectLoopImages operation."
        )


@register_message_handler("axis_image_response")
def axis_image_response(message: AxisImageResponseMessage, context: DhsContext):
    """Handles a single JPEG image from Axis video server for both getLoopTip
    and getLoopInfo operations."""

    _logger.debug(f"RECEIVED {message.file_length} BYTE IMAGE FROM AXIS VIDEO SERVER.")
    activeOps = context.get_active_operations()
    for ao in activeOps:
        if ao.operation_name == "getLoopTip" or "getLoopInfo":
            opName = ao.operation_name
            opHandle = ao.operation_handle
            image_key = ":".join([opName, opHandle])
            context.get_connection("automl_conn").send(
                AutoMLPredictRequest(image_key, message.file)
            )
        else:
            _logger.warning("RECEVIED JPEG, BUT NOT DOING ANYTHING WITH IT.")


def save_jpeg(image: bytes, index: int = None, save_dir: str = None):
    """
    Saves a JPEG image and increments the file number.
    e.g. if loop_0001.jpeg exists then the next file will be loop_0002.jpeg
    """
    new_num = index
    if new_num is None:
        current_images = glob.glob("".join([save_dir, "/*.jpeg"]))
        num_list = [0]
        for img in current_images:
            i = os.path.splitext(img)[0]
            try:
                num = re.findall("[0-9]+$", i)[0]
                num_list.append(int(num))
            except IndexError:
                pass
        num_list = sorted(num_list)
        new_num = num_list[-1] + 1

    save_name = "{}/loop_{:04}.jpeg".format(save_dir, new_num)

    f = open(save_name, "w+b")
    f.write(image)
    f.close()
    _logger.debug(f"SAVED JPEG IMAGE FILE: {save_name}")


def configure_logging(verbosity):

    loglevel = 20

    if verbosity >= 4:
        _logger.setLevel(logging.SPAM)
        loglevel = 5
    elif verbosity >= 3:
        _logger.setLevel(logging.DEBUG)
        loglevel = 10
    elif verbosity >= 2:
        _logger.setLevel(logging.VERBOSE)
        loglevel = 15
    elif verbosity >= 1:
        _logger.setLevel(logging.NOTICE)
        loglevel = 25
    elif verbosity <= 0:
        _logger.setLevel(logging.WARNING)
        loglevel = 30

    # verboselogs.install()

    logdir = "logs"

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logfile = os.path.join(logdir, Path(__file__).stem + ".log")
    handler = RotatingFileHandler(logfile, maxBytes=100000, backupCount=50)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(loglevel)
    _logger.addHandler(handler)

    # By default the install() function installs a handler on the root logger,
    # this means that log messages from your code and log messages from the
    # libraries that you use will all show up on the terminal.
    # coloredlogs.install(level='DEBUG')

    # If you don't want to see log messages from libraries, you can pass a
    # specific logger object to the install() function. In this case only log
    # messages originating from that logger will show up on the terminal.
    # coloredlogs.install(level='DEBUG', logger=logger)

    coloredlogs.install(
        level=loglevel,
        fmt="%(asctime)s,%(msecs)03d "
        + "%(hostname)s "
        + "%(name)s[%(funcName)s():%(lineno)d] "
        + "%(levelname)s "
        + "%(message)s ",
    )

    # LOG LEVELS AVAILABLE IN verboselogs module
    #  5 SPAM
    # 10 DEBUG
    # 15 VERBOSE
    # 20 INFO
    # 25 NOTICE
    # 30 WARNING
    # 35 SUCCESS
    # 40 ERROR
    # 50 CRITICAL

    # EXAMPLES
    # _logger.spam("this is a spam message")
    # _logger.debug("this is a debugging message")
    # _logger.verbose("this is a verbose message")
    # _logger.info("this is an informational message")
    # _logger.notice("this is a notice message")
    # _logger.warning("this is a warning message")
    # _logger.success("this is a success message")
    # _logger.error("this is an error message")
    # _logger.critical("this is a critical message")
    return logfile


def run():
    """Entry point for console_scripts."""

    main(sys.argv[1:])


def main(args):
    """Main entry point for allowing external calls."""

    dhs = Dhs()
    dhs.start()
    sigs = {}
    sigs = {signal.SIGINT, signal.SIGTERM}
    dhs.wait(sigs)


if __name__ == "__main__":
    run()
