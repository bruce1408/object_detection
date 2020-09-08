import os


root_dri = os.path.join(os.path.dirname(__file__), "..")
data_dir = os.path.join(root_dri, "data")


class imdb():
    def __init__(self, name):
        self._name = name
        self._classes = []
        self.__image_index = []
        self._roidb = None
        self._roidb_handle = self.default_roidb

    @property
    def name(self):
        return self._name

    def classes(self):
        return self._classes

    def roidb(self):
        if self._roidb is not None:
            return self.roidb
        self._roidb = self.roidb_handle
