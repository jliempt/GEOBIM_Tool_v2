from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import operator
import functools
import multiprocessing
import logging
from collections import defaultdict, Iterable, OrderedDict

# check pyqt version and import pyqt library
QString = str
os.environ['QT_API'] = 'pyqt5'
try:
    from pyqode.qt import QtCore
except BaseException:
    pass
from PyQt5 import QtCore, QtGui, QtWidgets

# add python-occ library
import OCC.Core.AIS
try:
    from OCC.Display.pyqt5Display import qtViewer3d
except BaseException:
    import OCC.Display
    try:
        import OCC.Display.backend
    except BaseException:
        pass
    try:
        OCC.Display.backend.get_backend("qt-pyqt5")
    except BaseException:
        OCC.Display.backend.load_backend("qt-pyqt5")
    from OCC.Display.qtDisplay import qtViewer3d

# add ifcopenshell library
import ifcopenshell
from ifcopenshell.geom.main import settings, iterator
from ifcopenshell.geom.occ_utils import display_shape,set_shape_transparency
from ifcopenshell import open as open_ifc_file
if ifcopenshell.version < "0.6":
    # not yet ported
    from .. import get_supertype

# add the functions of BIM calculation algorithm
from geobim.functions import *

def init():
    print("++++++++++++++++++++++++++")