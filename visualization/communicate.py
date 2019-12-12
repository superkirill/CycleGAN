import numpy as np
from torch import Tensor
from PyQt5.QtCore import pyqtSignal, QObject, QPoint


class Communicate(QObject):
    """Implement communication between classes"""
    # Signal requesting a redraw of the QGraphicsView
    # with the contents of the np.ndarray
    display_requested = pyqtSignal(np.ndarray, str, name="display")
    # Signal requesting random selection of
    # data for visualization
    random_requested = pyqtSignal(name="random")
    # Signal requesting fixed data for visualization
    # that is given in the Tensor
    fixed_requested = pyqtSignal(Tensor, str, name="fixed")
    # Signal sending the position of the mouse click in
    # the QGraphicsWidget
    graphics_clicked = pyqtSignal(QPoint)
    supress_whole_layer = pyqtSignal()
