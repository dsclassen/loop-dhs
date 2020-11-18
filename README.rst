========
loop-dhs
========


loop-dhs provides a command line tool **loopDHS** which is a python-based Distributed Hardware Server (DHS) built on the `pydhsfw <https://github.com/tetrahedron-technologies/pydhsfw>`_ package.

Description
===========

The loopDHS provides loop detection and classification functionality in support of fully automated macromolecular crystallography data collection.




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



