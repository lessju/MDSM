# Matplotlib stuff
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

# PyQt Stuff
import PyQt4 as Qt
import PyQt4.Qt as qt
import PyQt4.QtGui as gui
import PyQt4.QtCore as core
import PyQt4.uic as uic

# Python stuff
from scipy.signal import decimate
from math import ceil, sqrt
import struct, sys, os
import numpy as np

# ------------------------ MATPLOTLIB CLASS ---------------------------
class MatplotlibPlot:
    """ Class encapsulating an matplotlib plot """

    def __init__(self, parent = None, dpi = 100, size = (5,4)):
        """ Class initialiser """

        self.dpi = dpi
        self.figure = Figure(dpi = self.dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(parent)

        # Create the navigation toolbar, tied to the canvas
        self.toolbar = NavigationToolbar(self.canvas, parent)
        self.canvas.show()
        self.toolbar.show()

    def plotCurve(self, data, numCurves = 1, labels = None, xAxisRange = None, yAxisRange = None, xLabel = "", yLabel = ""):
        """ Normal plots """

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)        
        self.axes.grid(True)

        # Set empty labels if non defined
        if labels is None:
            labels = ['' for i in range(numImages)]

        # Place all plots in axes
        for i in range(numCurves):
            self.axes.plot(data[:,i], label=labels[i])
            self.axes.set_xlim((0, len(data[:,i])))

        # Final touches
        self.axes.xaxis.set_label_text(xLabel)
        self.axes.yaxis.set_label_text(yLabel)
        self.canvas.draw()

    def subplotCurve(self, data, numPlots = 1, labels = None, xAxisRange = None, yAxisRange = None, xLabel = "", yLabel = ""):
        """ Subplot mode """

        self.figure.clear()
  
        # Set empty labels if non defined
        if labels is None:
            labels = ['' for i in range(numImages)]

        if numPlots == 1:
            ax = self.figure.add_subplot(111)
            ax.plot(data[:])
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.set_title(labels[0])   
            ax.grid(True)         
        else:
            # Plot each image
            num_rows = ceil(sqrt(numPlots))
            for i in range(numPlots):
                ax = self.figure.add_subplot(num_rows, num_rows, i + 1)
                ax.plot(data[:,i])
                ax.set_xlim((0, len(data[:,i])))
                ax.set_xlabel(xLabel)
                ax.set_ylabel(yLabel)
                ax.set_title(labels[i])
                ax.grid(True)

        # Final touches
        self.figure.tight_layout()
        self.canvas.draw()


    def plotImage(self, data, numImages = 1, labels = None, xAxisRange = None, yAxisRange = None, xLabel = "", yLabel = ""):
        """ Image plot """

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)        

        # Show image
        im = self.axes.imshow(data, origin='lower', aspect='auto')
    
        # Final touches
        self.axes.xaxis.set_label_text(xLabel)
        self.axes.yaxis.set_label_text(yLabel)
        self.figure.colorbar(im)
        self.canvas.draw()

    def subplotImage(self, data, numImages = 1, labels = None, xAxisRange = None, yAxisRange = None, xLabel = "", yLabel = ""):
        """ Image subplot """

        # Clear figure
        self.figure.clear()

        # Set empty labels if non defined
        if labels is None:
            labels = ['' for i in range(numImages)]

        # Plot each image
        num_rows = ceil(sqrt(numImages))
        for i in range(numImages):
            ax = self.figure.add_subplot(num_rows, num_rows, i + 1)        
            im = ax.imshow(data[:,:,i],  origin='lower', aspect='auto')            
            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            ax.set_title(labels[i])

        # Final touches
        self.figure.tight_layout()
        self.canvas.draw()

# ------------------------- MAIN WINDOW --------------------------------
class BeamPlotter(gui.QMainWindow):
    """ Class encapsulating UI functionality"""

    def __init__(self, uiFile):
        """ Initialise main window """
        super(BeamPlotter, self).__init__()

        # Load window file
        self.mainWidget = uic.loadUi(uiFile)
        self.setCentralWidget(self.mainWidget)
        self.setWindowTitle("Beam Plotter")
        self.resize(920, 630)
        self.show()

        # Create menu bar
        self.menubar = self.menuBar()
        menu = self.menubar.addMenu("&File")
        openAct = menu.addAction('&Open')
        closeAct = menu.addAction('&Quit')
        self.statusBar()

        # Initialise matplotlib
        self.matlabPlot = MatplotlibPlot(self.mainWidget.plotFrame, 100)
        layout = gui.QGridLayout()
        self.mainWidget.plotFrame.setLayout(layout)
        layout.addWidget(self.matlabPlot.canvas)

        # Connect signals and slots
        core.QObject.connect(openAct, core.SIGNAL('triggered()'), self.openFile)
        core.QObject.connect(self.mainWidget.updateButton, core.SIGNAL('clicked()'), self.updateUI)
        core.QObject.connect(self.mainWidget.beamList, core.SIGNAL('itemSelectionChanged()'), self.beamsChanged)
        core.QObject.connect(self.mainWidget.typeCombo, core.SIGNAL('currentIndexChanged(int)'), self.plot)
        core.QObject.connect(self.mainWidget.subplotCheck, core.SIGNAL('clicked()'), self.plot)
        core.QObject.connect(self.mainWidget.logCheck, core.SIGNAL('clicked()'), self.logPressed)
        core.QObject.connect(self.mainWidget.timeSlider, core.SIGNAL('valueChanged(int)'), self.updateTime)
        core.QObject.connect(self.mainWidget.timeSlider, core.SIGNAL('sliderPressed()'), self.sliderPressed)
        core.QObject.connect(self.mainWidget.timeSlider, core.SIGNAL('sliderReleased()'), self.sliderReleased)
        core.QObject.connect(self.mainWidget.maskEdit, core.SIGNAL('editingFinished()'), self.setChannelMask)
        core.QObject.connect(self.mainWidget.intEdit, core.SIGNAL('editingFinished()'), self.applyIntegration)

        # Initialise
        self.filename = None
        self.mask = []
        self.nbeams = 1
        self.data = None

    # ====================================== Slider slots =============================
    def sliderPressed(self):
        self.moveTime = False

    def sliderReleased(self):
        self.moveTime = True
        self.updateTime()

    def updateTime(self):
        """ Update time infomration """

        if self.moveTime:
            self.sample = self.mainWidget.timeSlider.value()
            self.readDataSet()
            self.plot()
            self.mainWidget.timeLabel.setText("%.6f s" % (self.sample * self.tsamp))
        else:
            self.mainWidget.timeLabel.setText("%.6f s" % (self.mainWidget.timeSlider.value() * self.tsamp))

    def setChannelMask(self):
        """ Change channel mask """
        mask = str(self.mainWidget.maskEdit.text())
        mask = mask.split(',')
    
        self.mask = []
        for item in mask:
            try:
                dash = item.find('-')
                if dash >= 0:
                    beg, end = [int(val) for val in item.split('-')][:2]
                    self.mask.append([beg, end])
                else:
                    self.mask.append(int(item))
            except:
                continue
                
        self.mask = self.mask
        self.readDataSet()
        self.plot()

    def beamsChanged(self):
        """ Make sure that at least one beam is selected """
        if len(self.mainWidget.beamList.selectedItems()) == 0:
            self.statusBar().showMessage("At least one beam must be selected. Beam 0 chosen")
            self.mainWidget.beamList.item(0).setSelected(True)
        self.plot()

    def logPressed(self):
        """ Handle log """

        if self.data is None: return

        if self.mainWidget.logCheck.isChecked():
            self.data = 10 * np.log10(self.data)
            self.plot()
        else:
            self.readDataSet()
            self.plot()

    def applyIntegration(self):
        """ Apply new decimation factor """

        if self.filename is not None:
            self.integs = int(self.mainWidget.intEdit.text())
            self.readDataSet()
            self.plot()
        
    def updateInfo(self):
        """ Update data information """

        try:
            self.nchans      = int(self.mainWidget.chanEdit.text())
            self.fch1        = float(self.mainWidget.freqEdit.text())
            self.bw          = float(self.mainWidget.bwEdit.text())
            self.plotSamples = int(self.mainWidget.plotSamplesEdit.text())
            self.integs      = int(self.mainWidget.intEdit.text())
            self.sample      = self.mainWidget.timeSlider.value()
            self.log         = self.mainWidget.logCheck.isChecked()
            self.nbeams      = int(self.mainWidget.beamEdit.text())
            self.tsamp       = float(self.mainWidget.tsampEdit.text()) * self.integs
        except:
            self.statusBar().showMessage("Invalid data in information box")

        try:
            self.filesize = os.path.getsize(self.filename)
            self.num_samples = self.filesize / (self.nchans * self.nbeams * 4)
        except:
            pass

    def openFile(self):
        """ Start processing a new data file """

        dialog = gui.QFileDialog(self);
        dialog.setOptions(gui.QFileDialog.DontUseNativeDialog)
        dialog.setWindowTitle("Open data file")

        if dialog.exec_():
            if dialog.selectedFiles and dialog.selectedFiles():

                # Get file information
                self.filename = str(dialog.selectedFiles()[0])
            
                # Update UI information
                self.updateInfo()
                self.sample = 0

                # We have all required information, get data start plotting
                self.updateUI()
  

    def updateUI(self):
        """ Update UI """        

        # Update info
        self.updateInfo()

        # Populate beam list
        self.mainWidget.beamList.clear()

        for i in range(self.nbeams):
            item = gui.QListWidgetItem()
            item.setText("Beam %d" % (i + 1))
            item.setIndex = i
            self.mainWidget.beamList.addItem(item)

        # Set first beam as selected
        self.mainWidget.beamList.item(0).setSelected(True)

        # Update plots
        self.readDataSet()

        # Update slider
        self.mainWidget.timeSlider.setMaximum(self.num_samples - (self.plotSamples * self.integs))
        self.mainWidget.timeSlider.setValue(self.sample)

        # Update plot
        self.plot()


    def readDataSet(self):
        """ Read data set for further plotting """        

        # Seek to required position
        with open(self.filename, 'rb') as f:
            f.seek(self.sample * self.nchans * self.nbeams * 4, 0)
            self.data = f.read(self.plotSamples * self.integs * self.nchans * self.nbeams * 4)
            samples = len(self.data) / (self.integs * self.nchans * self.nbeams * 4)

            if samples == 0:
                return

            self.samples = samples

            # Shape data for plotting (downsample if necessary)
            self.data = np.array(struct.unpack('f' * self.samples * self.integs * self.nchans * self.nbeams, self.data))
            if self.integs <= 1:
                self.data = np.reshape(self.data, (self.samples * self.integs, self.nchans, self.nbeams))
            else:
                self.data = np.reshape(self.data, (self.integs, self.samples, self.nchans, self.nbeams))
                self.data = np.mean(self.data, axis=0)

            # Set log scale if required
            if self.mainWidget.logCheck.isChecked():
                self.data = 10 * np.log10(self.data)

            # Set channel mask if required
            if self.mask is not None and len(self.mask ) > 0:

                # Replace masked channel with interpolated value
                for i in range(self.nbeams):
                    bandpass = np.sum(self.data[:,:,i], axis=0) / self.plotSamples              
                    
                    # Apply to all masked channels
                    for item in self.mask:
                        # Check if applying mask to single channel or a range of channels
                        if isinstance(item, (tuple, list)):
                            value = bandpass[item[0]-2] + bandpass[item[0]-1] + bandpass[item[1]+1] + bandpass[item[0]+2]
                            self.data[:,item[0]:item[1],i] = np.ones((self.plotSamples, item[1]-item[0])) * (value * 0.25)
                        else:
                            value = bandpass[item-2] + bandpass[item-1] + bandpass[item+1] + bandpass[item+2]
                            self.data[:,item,i] = np.ones(self.plotSamples) * value * 0.25

            # All done
            self.statusBar().showMessage("Finished loading data")
            
            # Close processing dialog
            print "Done"
            dialog.close()


    def plot(self, index = 0):
        """ Main plotting function """

        # Check if we have data to plot
        try:
            if self.data is None:
                self.readDataSet()
        except:
            self.readDataSet()

        # Get plotting behaviour
        subplot = self.mainWidget.subplotCheck.isChecked()

        # Get list of beams which need to be plotted
        beams = np.array([self.mainWidget.beamList.row(item) for item in self.mainWidget.beamList.selectedItems()])

        # We have data, check which plot type we require
        index = self.mainWidget.typeCombo.currentIndex()

        # ============================== Waterfall plot =================================
        if index == 0:
            if subplot:
                labels = ["Beam %d" % (i + i) for i in beams]
                self.matlabPlot.subplotImage(self.data[:,:,beams], 
                                             numImages=len(beams), 
                                             labels=labels,  
                                             xLabel='Frequency', 
                                             yLabel='Time')
            else:
                self.matlabPlot.plotImage(self.data[:,:,beams[0]], xLabel='Frequency', yLabel='Time')                

        # ============================== Timeseries plot =================================
        elif index == 1:
            labels = ["Beam %d" % (i + i) for i in beams]
            if subplot:
                self.matlabPlot.subplotCurve((np.sum(self.data, axis=1) / self.nchans)[:,beams], 
                                             numPlots = len(beams), 
                                             labels = labels, 
                                             xLabel = "Time", 
                                             yLabel = "Arbitrary Power")
                
            else:
                self.matlabPlot.plotCurve((np.sum(self.data, axis=1) / self.nchans)[:,beams], 
                                          numCurves = len(beams), 
                                          labels = labels, 
                                          xLabel = "Time", 
                                          yLabel = "Arbitrary Power")

        # ============================== Bandpass plot =================================
        else:
            labels = ["Beam %d" % (i + i) for i in beams]
            if subplot:
                self.matlabPlot.subplotCurve((np.sum(self.data, axis=0) / self.plotSamples)[:,beams], 
                                             numPlots = len(beams), 
                                             labels = labels, 
                                             xLabel = "Frequency", 
                                             yLabel = "Arbitrary Power")
                
            else:
                self.matlabPlot.plotCurve((np.sum(self.data, axis=0) / self.plotSamples)[:,beams], 
                                          numCurves = len(beams), 
                                          labels = labels, 
                                          xLabel = "Frequency", 
                                          yLabel = "Arbitrary Power")


# Application entry point
if __name__ == "__main__":
    app = gui.QApplication(sys.argv)
    app.setApplicationName("BeamPlotter")
    plotter = BeamPlotter("beamPlotter.ui")
    sys.exit(app.exec_())
