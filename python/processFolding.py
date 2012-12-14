# PyQt Stuff
import PyQt4 as Qt
import PyQt4.Qt as qt
import PyQt4.QtGui as gui
import PyQt4.QtCore as core
import PyQt4.uic as uic
import PyQt4.Qwt5 as Qwt

# Matplotlib stuff
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

# Other stuff
import numpy as np
import struct
import sys, os

# ------------------------ MATPLOTLIB CLASS ---------------------------

class MatplotlibPlot:
    """ Class encapsulating an matplotlib plot"""
    def __init__(self, parent = None, dpi = 100, size = (5,4)):
        """ Class initialiser """

        self.dpi = dpi
        self.figure = Figure(size, dpi = self.dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(parent)

        # Create the navigation toolbar, tied to the canvas
        self.toolbar = NavigationToolbar(self.canvas, parent)
        self.canvas.show()
        self.toolbar.show()

    def plotCurve(self, data, xAxisRange = None, yAxisRange = None, xLabel = "", yLabel = ""):
        """ Plot the data as a curve"""

        # clear the axes and redraw the plot anew
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)        
        self.axes.grid(True)
        self.axes.plot(range(np.size(data)), data)

        if xAxisRange is not None:        
            self.xAxisRange = xAxisRange
            self.axes.xaxis.set_major_formatter(ticker.FuncFormatter(
                       lambda x, pos=None: '%.2f' % self.xAxisRange[x] if 0 <= x < len(xAxisRange) else ''))
            for tick in self.axes.xaxis.get_ticklabels():
                  tick.set_rotation(15)

        if yAxisRange is not None:        
            self.yAxisRange = yAxisRange
            self.axes.xaxis.set_major_formatter(ticker.FuncFormatter(
                       lambda x, pos=None: '%.1f' % self.yAxisRange[y] if 0 <= y < len(yAxisRange) else ''))
            for tick in self.axes.yaxis.get_ticklabels():
                  tick.set_rotation(15)

        self.axes.xaxis.set_label_text(xLabel)
        self.axes.yaxis.set_label_text(yLabel)
        self.canvas.draw()

# ------------------------- MAIN WINDOW --------------------------------

class UIWindow(gui.QMainWindow):
    """ Main UI Window class """

    def __init__(self, uiFile):
        """ Initialise main window """
        super(UIWindow, self).__init__()

        # Load window file
        self.mainWidget = uic.loadUi(uiFile)
        self.setCentralWidget(self.mainWidget)
        self.setWindowTitle("Folded Output Plotter")
        self.resize(610,690)

        # Create menu bar
        self.menubar = self.menuBar()
        menu = self.menubar.addMenu("&File")
        openAct = menu.addAction('&Open')
        closeAct = menu.addAction('&Quit')
        self.statusBar().showMessage("Open file containing folded output")

        # Create matplotlib Figure and FigCanvas objects
        self.foldedPlot = MatplotlibPlot(self.mainWidget.plotWidget, 100, (6,6))

        # File watcher
        self.watcher = core.QFileSystemWatcher(self)
        
        # Connect signals and slots
        core.QObject.connect(openAct, core.SIGNAL('triggered()'), self.openFile)
        core.QObject.connect(closeAct, core.SIGNAL('triggered()'), self.quit)
        core.QObject.connect(self.mainWidget.dmSlider, core.SIGNAL('valueChanged(int)'), self.plot)
        core.QObject.connect(self.watcher, core.SIGNAL("fileChanged(const QString&)"), self.loadFile)

        self.show()

    def openFile(self):
        """ Start processing a output file """
        dialog = gui.QFileDialog(self);
        dialog.setOptions(gui.QFileDialog.DontUseNativeDialog)
        dialog.setWindowTitle("Open file with folded output")

        if dialog.exec_() :
            if dialog.selectedFiles and dialog.selectedFiles():
                for filename in self.watcher.files():
                    self.watcher.removePath(filename)

                self.loadFile(dialog.selectedFiles()[0])
                self.watcher.addPath(dialog.selectedFiles()[0])
            else:
                self.statusBar().showMessage("No files selected...", 2000)
                return

    def loadFile(self, filename):
        """ Load file """
        self.statusBar().showMessage("Processing output file!")
        
        with open(filename, 'rb') as f:
            # Read header
            data = f.read(20)
            self._tdms, self._dmstep, self._tsamp, self._period, self._bins = struct.unpack('ifffi', data)

            # Process data
            filesize = os.path.getsize(filename) - 20
            buffSize = self._tdms * self._bins;
            self._data = np.zeros((self._tdms, self._bins), dtype='float')

            for i in range(filesize / (buffSize * 4)):
                data = f.read(buffSize * 4)
                tempData = np.array(struct.unpack(buffSize * 'f', data))
                tempData = tempData.reshape(self._tdms, self._bins)
                self._data = self._data + tempData

        # Update UI
        self.mainWidget.dmSlider.setMaximum(self._tdms - 1)
        self.plot()

        self.statusBar().showMessage("File processed successfully", 2000)

    def plot(self, someArg = None):
        """ Updates the plots """

        val = self.mainWidget.dmSlider.value()

        self.mainWidget.dmSlider.maximum = self._tdms
        self.mainWidget.dmLabel.setText("DM: %.3f" % (val * self._dmstep))
        self.foldedPlot.plotCurve(self._data[val], xLabel = "Folded Time")


    def savePlot(self):
        """ Save plot to file """
        file_choices = "PNG (*.png)|*.png"
        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.canvas.print_figure(path, dpi = self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

    def quit(self):
        """ Quit application """
        exit()

if __name__ == "__main__":

    if len(sys.argv) == 3:

        # Process file offline
        with open(sys.argv[1], 'rb') as f:
            print "Processing file..."

            # Read header
            header = f.read(20)
            tdms, dmstep, tsamp, period, bins = struct.unpack('ifffi', header)

            # Process data
            filesize = os.path.getsize(sys.argv[1]) - 20
            buffSize = tdms * bins;
            data = np.zeros((tdms, bins), dtype='float')

            for i in range(filesize / (buffSize * 4)):
                read_data = f.read(buffSize * 4)
                tempData = np.array(struct.unpack(buffSize * 'f', read_data))
                tempData = tempData.reshape(tdms, bins)
                data = data + tempData

            # Dump processed data to file
            with open(sys.argv[2], 'wb') as f:
                f.write(struct.pack('ifffi', tdms, dmstep, tsamp, period, bins))
                for i in range(tdms):
                    for j in range(bins):
                        f.write(struct.pack('f', data[i,j]))

    elif len(sys.argv) == 1:
        # Process file through UI
        app = gui.QApplication(sys.argv)
        app.setApplicationName("FoldedPlotter")
        window = UIWindow("processFolding.ui")
        sys.exit(app.exec_())

    else:
        print "Script requried either 2 arguments (offline) or none, %d provided" % (len(sys.argv) - 1)
