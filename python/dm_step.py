from PyQt4.Qwt5 import QwtPlot
import PyQt4.Qwt5 as Qwt
from PyQt4 import Qt
from math import fabs, sqrt
import numpy as np
import sys

class DispersionStepPlot(Qwt.QwtPlot):
    """ Dispersion plot widget, containing dipersion parameters """

    def __init__(self, freq, bandwidth, pulseWidth, tsamp = 0, useSNR=True):
        """ Class constructor """

        # Call base class chain constructors
        super(DispersionStepPlot, self).__init__()

        # Input parameters
        self.freq = freq
        self.bandwidth = bandwidth
        self.pulseWidth = pulseWidth
        self.useSNR = useSNR
        self.tsamp = tsamp
        self.centerDm = 50

        # Initialise class
        self._reset_ploting_area()

    # Define class properties (getters and setter are defined in the lambda functions)
    freq = property(lambda self: self._freq, lambda self, val: setattr(self, "_freq", float(val)))
    bandwidth = property(lambda self: self._bandwidth, lambda self, val: setattr(self, "_bandwidth", float(val)))
    pulseWidth = property(lambda self: self._pulseWidth, lambda self, val: setattr(self, "_pulseWidth", float(val)))
    tsamp = property(lambda self: self._tsamp, lambda self, val: setattr(self, "_tsamp", float(val)))
    useSNR = property(lambda self: self._useSNR, lambda self, val: setattr(self, "_useSNR", val))

    def effective_width(self, intrinsic_width, dm, bandwidth, freq):
        """ Calculate the effecrtive pulse width """
        a = sqrt(pow(intrinsic_width, 2) + pow((8.3e6 * fabs(dm) * (bandwidth / pow(freq, 3))), 2))
        return a
        
    def effective_snr(self, effective_width, period):
        """ Calculate the effective SNR """
        return sqrt((period - effective_width) / (effective_width * 1.0)) if period > effective_width else 0


    def _reset_ploting_area(self):
        """" Resets the plotting area """

        # Clear any existing curves and markers
        self.clear()

        # Initialise canvas
        self.setTitle('Disperion Measure Step Plot')
        self.setCanvasBackground(Qt.Qt.white)
        self.plotLayout().setMargin(0)
        self.plotLayout().setCanvasMargin(0)
        self.plotLayout().setAlignCanvasToScales(True)
        self.setAxisTitle(QwtPlot.yLeft, 'Signal to Noise Ratio')
        self.setAxisTitle(QwtPlot.xBottom, 'Trial Dispersion Measure')

        # Attached Grid
        grid = Qwt.QwtPlotGrid()
        grid.attach(self)
        grid.setPen(Qt.QPen(Qt.Qt.black, 0, Qt.Qt.DotLine))

    def _calculate_snr_spread(self):
        """ Calculate the SNR spread """

        dmSpacing, percentage = 100, 0
        while percentage < 0.5:        
            x = np.linspace(self.centerDm - dmSpacing, self.centerDm + dmSpacing, 500)
            y = np.array([self.effective_snr(self.effective_width(self.pulseWidth, self.centerDm - dm_val, self.bandwidth, self.freq), self.pulseWidth * 20) for dm_val in x])
            y = (y / (np.max(y) * 1.0)) if np.max(y) > 0 else y
            percentage = np.size(np.where(y > 0)) / 1000.0
            dmSpacing = dmSpacing*0.6
    
        return x, y

    def calculate_optimal_dmstep(self, acceptedSNR= 95):
        """ Calculate the optimal DM step, within the specified accepted SNR percentage """

        if not self.useSNR:
            return 1.205e-7 * self.tsamp * (self.freq ** 3) / self.bandwidth
       
        x, y = self._calculate_snr_spread()
        return fabs(self.centerDm - x[np.max(np.where(y > np.max(y) * float(acceptedSNR) / 100.0 ))])
        
  
    def new_plot_dmstep(self):
        """ Create the dispersion plot on a clear canvas """
        self._reset_ploting_area()
        self.plot_dmstep()

    def plot_dmstep(self):
        """ Create the dispersion plot """

        x, y = self._calculate_snr_spread()

        # Attach a curve
        curve = Qwt.QwtPlotCurve()
        curve.attach(self)
        curve.setPen(Qt.QPen(Qt.Qt.blue, 2))
        curve.setData(x, y)
     
        self.replot()

if __name__ == "__main__":
    """ Executing as a main script """
    app = Qt.QApplication(sys.argv)
    step = DispersionStepPlot(30.0, 6, 1)
    step.plot_dmstep()
    step.resize(600, 500)
    step.show()
    sys.exit(app.exec_())
