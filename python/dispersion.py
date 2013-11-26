from PyQt4.Qwt5 import QwtPlot
import PyQt4.Qwt5 as Qwt
from PyQt4 import Qt
import numpy as np
import sys

class DispersionPlot(Qwt.QwtPlot):
    """ Dispersion plot widget, containing dipersion parameters """

    def __init__(self, highFreq, freqOffset, nchans, pulseWidth = 1, maximumSmearing = 100):
        """ Class constructor """

        # Call base class chain constructors
        super(DispersionPlot, self).__init__()

        # Input parameters
        self.highFreq = highFreq
        self.freqOffset = freqOffset
        self.nchans = nchans
        self.pulseWidth = pulseWidth
        self.maximumSmearing = maximumSmearing

        # Initialise class
        self._reset_ploting_area()

    # Define class properties (getters and setter are defined in the lambda functions)
    highFreq = property(lambda self: self._highFreq, lambda self, val: setattr(self, "_highFreq", float(val)))
    freqOffset = property(lambda self: self._freqOffset, lambda self, val: setattr(self, "_freqOffset", float(val)))
    nchans = property(lambda self: self._nchans, lambda self, val: setattr(self, "_nchans", val))
    pulseWidth = property(lambda self: self._pulseWidth, lambda self, val: setattr(self, "_pulseWidth", float(val)))
    maximumSmearing = property(lambda self: self._maximumSmearing, lambda self, val: setattr(self, "_maximumSmearing", float(val)))

    def total_dispersion(self):
        """ Calculate the total dispersion within the bandwidth for a DM of 0 """
        return 4.15e6 * ((self.highFreq - self.freqOffset * self.nchans)**-2 - self.highFreq**-2)

    def channel_dispersion(self):
        """ Calculate the dispersion smearing within a channel for a DM of 0"""
        return 8.3e6 * self.freqOffset *  (self.highFreq - self.freqOffset * self.nchans / 2.0)**-3

    def _reset_ploting_area(self):
        """" Resets the plotting area """
        
        # Clear any existing curves and markers
        self.clear()

        # Initialise canvas
        self.setTitle('Dispersion Curve')
        self.setCanvasBackground(Qt.Qt.white)
        self.plotLayout().setMargin(0)
        self.plotLayout().setCanvasMargin(0)
        self.plotLayout().setAlignCanvasToScales(True)
        self.setAxisTitle(QwtPlot.yLeft, 'Frequence (MHz)')
        self.setAxisTitle(QwtPlot.xBottom, 'Dispersion Time (ms)')

        # Attached Grid
        grid = Qwt.QwtPlotGrid()
        grid.attach(self)
        grid.setPen(Qt.QPen(Qt.Qt.black, 0, Qt.Qt.DotLine))
  
    def new_plot_scattering(self):
        """ Create the dispersion plot on a clear canvas """
        self._reset_ploting_area()
        self.plot_scattering()

    def plot_dispersion(self):
        """ Create the dispersion plot """

        # Calculate curve
        y = np.linspace(self.highFreq - (self.nchans * self.freqOffset), self.highFreq - self.freqOffset, 100)
        x = 4.15e6 * (y**-2 - self.highFreq**-2)

        # Attach a curve
        curve = Qwt.QwtPlotCurve()
        curve.attach(self)
        curve.setPen(Qt.QPen(Qt.Qt.blue, 2))
        curve.setData(x, y)
     
        self.replot()

if __name__ == "__main__":
    """ Executing as a main script """
    app = Qt.QApplication(sys.argv)
    disp = DispersionPlot(30, 0.195, 31)
    disp.plot_dispersion()
    disp.resize(600, 500)
    disp.show()
    sys.exit(app.exec_())
