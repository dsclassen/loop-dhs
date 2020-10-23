========
loop-dhs
========


loop-dhs provides a command line tool, loopDHS, which is a python-based Distributed Hardware Server (DHS) built on the `pydhsfw <https://github.com/tetrahedron-technologies/pydhsfw>`_ package.

Description
===========

The loopDHS provides loop detection and classification functionality in support of fully automated macromolecular crystallography data collection.

will communicate with an instance of DCSS


Installation
============

requirements:
virtualenv
python 3.8 (might work on 3-6-3.7, I haven't tested)
pydhsfw

....

AXIS Video Receiver Port  
==========================

For a loopDHS we will need to open a port than can receive a stream of jpeg images from our axis video server. The AutoML API requires that images be base64 encoded.

....

RESTful API loop detection and classification  
===============================================

loopDHS is currently configured to use Google Cloud Platform (GCP) AutoML vision to generate a deep learning vision model.

1. Train and Download a loop classification and detection model.
2. Configure a local GPU machine with docker and GCP docker image.
3. Test the REST API.

Details of the Google Cloud AutoML docker stuff will go here.  

* Overview of `AutoML <https://cloud.google.com/automl>`_ tools on Google Cloud Platform.
* `Details <https://cloud.google.com/vision/automl/docs/edge-quickstart>`_ for training an AutoML Vision model.
* `Tutorial <https://cloud.google.com/vision/automl/docs/containers-gcs-tutorial>`_ to deploy your model in an Edge container.



....

These are all the operations the current camera DHS is responsible for  
========================================================================

.. code-block:: sh

   initializeCamera  
   getLoopTip  
   getPinDiameters
   addImageToList
   findBoundingBox
   getVerticalCut
   getLoopInfo
   collectLoopImages
   stopCollectLoopImages
   reboxLoopImage


We may not need/want all of these in new loopDHS

....

Psuedo code for a loop DHS
==========================

`loopFast.tcl` or similar scripted operation running in the dcss tcl interpreter performs the following:  

.. code-block:: sh

   dcss/loopFast sends collectLoopImages to loopDHS (stoh_start_operation )  
      loopDHS starts listening for jpg images via http socket from axis server  
   dcss/loopFast start the gonio moving via a `start_oscillation gonio_phi video_trigger $osci_delta $osci_time`  
      loopDHS is receiving the jpegs and storing them somehow.  
   dcss/loopFast sends stopCollectLoopImages  
      loopDHS sends images to docker for loop classification and detection.  
      loopDHS does some minimal set of calculation from the bbox data received from docker.  
      loopDHS returns a list of list. we can discuss exactly what gets passed back.  


I'm pretty sure there is a 1024 byte limit to each ``xos2`` response so we will probably have to break this down and send the results from each image back to DCSS one at a time, and then reassemble within the ``loopFast.tcl`` scripted operation.

.. code-block:: tcl

   [
   [image_num, tipX, tipY, bboxMinX, bboxMaxX, bboxMinY, bboxMaxY, loop_width, loop_type],
   [image_num, tipX, tipY, bboxMinX, bboxMaxX, bboxMinY, bboxMaxY, loop_width, loop_type],
   .
   .
   .
   [image_num, tipX, tipY, bboxMinX, bboxMaxX, bbpxMinY, bboxMaxY, loop_width, loop_type],
   ]

....


Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
