# PyQt Stuff
import PyQt4 as Qt
import PyQt4.Qt as qt
import PyQt4.QtGui as gui
import PyQt4.QtCore as core
import PyQt4.uic as uic

# Matplotlib stuff
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

# Other stuff
import numpy as np, math
import struct, re
import sys, os
import scipy

num_samples = 500

# =========================== File helper methods ===================================
def openFile(path):
    """ Open file and return pointer and data"""
    f = open(path, 'rb')

    # Extract info
    string = f.readline()
    data = re.match("nchans=(?P<nchans>\d+),nbeams=(?P<nbeams>\d+),tsamp=(?P<tsamp>\d+\.\d+).*", string)
    if data == None:
        print "Incorrect data format"
        exit()
    info = data.groupdict()

    info['nbeams'] = int(info['nbeams'])
    info['nchans'] = int(info['nchans'])
    info['tsamp'] = float(info['tsamp'])
    info['header_len'] = len(string)
    info['nsamp'] = (os.path.getsize(path) - info['header_len']) / (4 * info['nchans'] * info['nbeams'])
    info['fp'] = f

    return info


# ============================ Matplotlib Class =====================================
class MatplotlibPlot:
    """ Class encapsulating a matplotlib plot"""

    def __init__(self, parent = None, dpi = 100, size = (5,5)):
        """ Class initialiser """

        self.dpi = dpi
        self.figure = Figure(size, dpi = self.dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(parent)

        # Create the navigation toolbar, tied to the canvas
        self.toolbar = NavigationToolbar(self.canvas, parent)
        self.canvas.show()
        self.toolbar.show()

        # Reset the plot landscape
        self.figure.clear()

    def plotMultiPixel(self, info, data):
        """ Generate multi-pixel plot """

        # Tabula Rasa
        self.figure.clear()
        rows = math.ceil(math.sqrt(info['nbeams']))

	    # Display a subplot per beam (randomly for now)
        for i in range(info['nbeams']):
            ax = self.figure.add_subplot(rows, rows, i)
            ax.plot(data[:,512,i])
            
        

    def updatePlot(self):
        self.canvas.draw()

# ============================ Main Window Class =====================================
class MultiPixelPlotter(gui.QMainWindow):
    """ Main UI Window class """

    def __init__(self, uiFile):

        """ Initialise main window """
        super(MultiPixelPlotter, self).__init__()

        # Load window file
        self.mainWidget = uic.loadUi(uiFile)
        self.setCentralWidget(self.mainWidget)
        self.setWindowTitle("Nice Plotter")
        self.resize(880,740)

        # Create matplotlib Figure and FigCanvas objects
        self.matlabPlot = MatplotlibPlot(self.mainWidget.plotFrame, 100, (6,6))
        layout = gui.QGridLayout()
        self.mainWidget.plotFrame.setLayout(layout)
        layout.addWidget(self.matlabPlot.canvas)

        # Connect signals and slots
        core.QObject.connect(self.mainWidget.timeSlider, core.SIGNAL('valueChanged(int)'), self.timeChanged)

        # Load input file
        if len(sys.argv) < 2:
            print "Input file required"
            exit()

        self.info = openFile(sys.argv[1])    

        # Update slider and UI information
        self.mainWidget.infoLabel.setText("Number of Beams:   %d\nNumber of channels:   %d\nNumber of samples:   %d" % 
                                         (self.info['nbeams'], self.info['nchans'], self.info['nsamp']))
        self.mainWidget.timeSlider.setRange(0, self.info['nsamp'])
        self.show()

        # Update plot
        self.plot()


    def plot(self):
        """ Populate plot """
        nbeams = self.info['nbeams']
        nchans = self.info['nchans']
        nsamp  = self.info['nsamp']
        fp     = self.info['fp']

        # Get plot information
        integrations = self.mainWidget.intSpin.value()
        sample = self.mainWidget.timeSlider.value()

        # Rewind file to required location
        fp.seek(self.info['header_len'] + sample * integrations * nchans * nbeams * 4)

        # Read required data
        data = fp.read( num_samples * integrations * nchans * nbeams * 4 )
        data = struct.unpack('f' * num_samples * integrations * nchans * nbeams, data)
        data = np.reshape(data, (num_samples * integrations, nchans, nbeams))
        
        # Decimate data if required 
        if integrations > 1:
            scipy.signal.decimate(data, integrations, axis = 0)
        
        self.matlabPlot.plotMultiPixel(self.info, data)
        
    def timeChanged(self, time):
        """ Time slider value changed """
        self.plot()


# Script entry point
if __name__ == "__main__":
    app = gui.QApplication(sys.argv)
    app.setApplicationName("Multi-Pixel Plotter")
    window = MultiPixelPlotter("plotter.ui")
    sys.exit(app.exec_())
