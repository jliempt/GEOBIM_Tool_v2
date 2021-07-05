import sys

from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "PyQt5 Drawing Tutorial"
        self.top = 150
        self.left = 150
        self.width = 500
        self.height = 500
        # self.topleftdockwindow()
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)

        # parcel boundary
        painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
        painter.drawRect(20, 100, 450, 220)
        # floor boundary
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.gray, Qt.SolidPattern))
        painter.drawRect(40, 90, 400, 250)

        # line check
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        painter.drawLine(40, 340, 440, 340)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawLine(40, 90, 440, 90)
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter.drawLine(40, 90, 40, 340)
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter.drawLine(440, 90, 440, 340)
        # road polygons
        painter.setPen(QPen(Qt.blue, 4, Qt.SolidLine))
        painter.drawLine(0, 400, 600, 400)
        painter.setPen(QPen(Qt.blue, 4, Qt.SolidLine))
        painter.drawLine(0, 50, 600, 50)
        # road names
        font = QtGui.QFont()
        font.setFamily('Times')
        font.setBold(True)
        font.setPointSize(15)
        painter.setFont(font)
        painter.drawText(0, 410, 600, 550, Qt.AlignHCenter, 'Bompjes')
        painter.drawText(0, 20, 600, 40, Qt.AlignHCenter, 'Hertekade')

    def topleftdockwindow(self):
        self.items = QDockWidget('Dockable', self)

        self.listWidget = QListWidget()
        self.listWidget.addItem('Boompjes')
        self.listWidget.addItem('   Pass')
        self.listWidget.addItem('   Admissible overhang: 1.5')
        self.listWidget.addItem('   Overhang: 1')
        self.listWidget.addItem('Hertekade')
        self.listWidget.addItem('   Fail')
        self.listWidget.addItem('   Admissible overhang: 0.2')
        self.listWidget.addItem('   Overhang: 0.5')

        self.items.setWidget(self.listWidget)
        self.items.setFloating(False)
        self.setCentralWidget(QTextEdit())
        self.addDockWidget(Qt.RightDockWidgetArea, self.items)
        self.setWindowTitle('Dock')


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())

