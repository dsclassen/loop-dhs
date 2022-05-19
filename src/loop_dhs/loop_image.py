class LoopImageSet:
    """Class to hold set of JPEG images acquired via collectLoopImages operation.

    Attributes:
        images (list): List of images sent from AXIS video server
        results (list): List of results from AutoML

    """

    def __init__(self):
        """Constructor method
        """
        self.images = []
        self.results = []
        self._number_of_images = None

    def add_image(self, image: bytes):
        """Add a jpeg image to the list of images.
        """
        self.images.append(image)
        self._number_of_images = len(self.images)

    def add_results(self, result: list):
        """
        Add the AutoML results to a list for use in reboxLoopImage
        loop_info stored as python list so this is a list of lists.
        """
        self.results.append(result)

    @property
    def number_of_images(self) -> int:
        """Get the number of images in the image list"""
        return self._number_of_images


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
