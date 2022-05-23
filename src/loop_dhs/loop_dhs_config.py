
import os
from datetime import datetime as dt
from dotty_dict.dotty_dict import Dotty
from loop_dhs.loop_image import LoopImageSet

class LoopDHSConfig(Dotty):
    """Class to wrap DHS configuration settings."""

    def __init__(self, conf_dict: dict):
        super().__init__(conf_dict)
        self.timestamped_debug_dir = None

    def make_debug_dir(self):
        now = dt.now().strftime("%Y-%m-%d-%H%M%S")
        self.timestamped_debug_dir = os.path.join(self['loopdhs.debug_dir'], now)
        os.makedirs(self.timestamped_debug_dir)


    @property
    def dcss_url(self):
        return 'dcss://' + str(self['dcss.host']) + ':' + str(self['dcss.port'])

    @property
    def automl_url(self):
        return (
            'http://'
            + str(self['loopdhs.automl.host'])
            + ':'
            + str(self['loopdhs.automl.port'])
        )

    @property
    def jpeg_receiver_url(self):
        return 'http://localhost:' + str(self['loopdhs.jpeg_receiver.port'])

    @property
    def axis_url(self):
        return (
            'http://'
            + str(self['loopdhs.axis.host'])
            + ':'
            + str(self['loopdhs.axis.port'])
        )

    @property
    def axis_camera(self):
        return self['loopdhs.axis.camera']

    @property
    def save_images(self):
        return self['loopdhs.save_image_files']

    @property
    def debug_dir(self):
        return self['loopdhs.debug_dir']

    @property
    def log_dir(self):
        return self['loopdhs.log_dir']

    @property
    def automl_thhreshold(self):
        return self['loopdhs.automl.threshold']

    @property
    def automl_scan_n_results(self):
        return self['loopdhs.automl.scan_n_results']

    @property
    def osci_delta(self):
        return self['loopdhs.osci_delta']

    @property
    def osci_time(self):
        return self['loopdhs.osci_time']

    @property
    def video_fps(self):
        return self['loopdhs.video_fps']


class LoopDHSState:
    """Class to hold DHS state info."""

    def __init__(self):
        self._rebox_images = None
        self._collect_images = False

    @property
    def rebox_images(self) -> LoopImageSet:
        return self._rebox_images

    @rebox_images.setter
    def rebox_images(self, images: LoopImageSet):
        self._rebox_images = images

    @property
    def collect_images(self) -> bool:
        return self._collect_images

    @collect_images.setter
    def collect_images(self, collect: bool):
        self._collect_images = collect


