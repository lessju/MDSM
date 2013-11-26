from PyQt4.Qwt5 import QwtPlot
import PyQt4.Qwt5 as Qwt
from PyQt4 import Qt
from math import log10, sqrt
import numpy as np
import sys

class ScatteringPlot(Qwt.QwtPlot):
    """ Dispersion plot widget, containing dipersion parameters """

    def __init__(self, freq, bandwidth, pulseWidth, allowedScattering = 500):
        """ Class constructor 
            freq = Frequency in MHz
            pulseWidth = minimum pulse widths in ms """

        # Call base class chain constructors
        super(ScatteringPlot, self).__init__()

        # Input parameters
        self.freq = freq
        self.pulseWidth = pulseWidth
        self.bandwidth = bandwidth
        self.allowedScattering = allowedScattering

        # Initialise class
        self._reset_ploting_area()

    # Define class properties (getters and setter are defined in the lambda functions)
    freq = property(lambda self: self._freq, lambda self, val: setattr(self, "_freq", val / 1000.0))
    bandwidth = property(lambda self: self._bandwidth, lambda self, val: setattr(self, "_bandwidth", val))
    pulseWidth = property(lambda self: self._pulseWidth, lambda self, val: setattr(self, "_pulseWidth", float(val)))
    allowedScattering = property(lambda self: self._allowedScattering, lambda self, val: setattr(self, "_allowedScattering", val / 100.0))

    def __calculate_dm(self, delay, frequency):
        """ Calculate the empirical DM associated with a givan frequency and tolerated delay """
        freq = self.freq - self.bandwidth / 2.0
        try:
            b = log10(delay)
            c = -6.46 - 3.86 * log10(frequency) - b
            d = (0.154 ** 2)  - (4 * 1.07 * c)
            ldm1, ldm2 = (-0.54 + sqrt(d)) / 2* 1.07, (-0.54 - sqrt(d)) / 2* 1.07
            return pow(10, ldm1), pow(10, ldm2)
        except:
            return 0, 0

    def calculate_dm(self):
        """ Calculate empirical DM associated with initialised parameters """
        return self.__calculate_dm(self.pulseWidth * self.allowedScattering, self.freq)

    def _reset_ploting_area(self):
        """" Resets the plotting area """

        # Clear any existing curves and markers
        self.clear()

        # Initialise canvas
        self.setTitle('Scattering Curve')
        self.setCanvasBackground(Qt.Qt.white)
        self.plotLayout().setMargin(0)
        self.plotLayout().setCanvasMargin(0)
        self.plotLayout().setAlignCanvasToScales(True)
        self.setAxisTitle(QwtPlot.yLeft, 'Dispersion Measure')
        self.setAxisTitle(QwtPlot.xBottom, 'Scattering Timescale (ms)')

        # Attached Grid
        grid = Qwt.QwtPlotGrid()
        grid.attach(self)
        grid.setPen(Qt.QPen(Qt.Qt.black, 0, Qt.Qt.DotLine))
  
    def new_plot_scattering(self):
        """ Create the dispersion plot on a clear canvas """
        self._reset_ploting_area()
        self.plot_scattering()

    def plot_scattering(self):
        """ Create the dispersion plot """

        # Calculate curve
        x = np.linspace(0.01, max(self.pulseWidth * 5, self.pulseWidth * self.allowedScattering), 100)
        y = np.array([max(self.__calculate_dm(xval, self.freq)) for xval in x])

        # Attach a curve
        curve = Qwt.QwtPlotCurve()
        curve.attach(self)
        curve.setPen(Qt.QPen(Qt.Qt.blue, 2))
        curve.setData(x, y)
     
        self.replot()

if __name__ == "__main__":
    """ Executing as a main script """
    app = Qt.QApplication(sys.argv)
    scat = ScatteringPlot(40, 6, 1)
    scat.plot_scattering()
    scat.resize(600, 500)
    scat.show()
    sys.exit(app.exec_())
