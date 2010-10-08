import elementtree.ElementTree as ET
from math import ceil, log, pow
import PyQt4
import PyQt4.QtGui as gui
import PyQt4.QtCore as core
import PyQt4.uic as uic
import sys

from dispersion import *
from scattering import *
from dm_step import *
import ddplan as presto


""" Main MDSM Parameters script """

class MainWindow(gui.QMainWindow):

    def __init__(self, uiFile):
        """ Initialise Window """

        super(MainWindow, self).__init__()

        # Load window file        
        self.mainWidget = uic.loadUi(uiFile)
        self.setCentralWidget(self.mainWidget)
        self.setWindowTitle("MDSM Parameters")
        self.resize(890, 585)
        self.show()

        # Connect signals and slots 
        core.QObject.connect(self.mainWidget.resetButton, core.SIGNAL('clicked()'), self.reset)
        core.QObject.connect(self.mainWidget.calcButton, core.SIGNAL('clicked()'), self.calculate)
        core.QObject.connect(self.mainWidget.saveButton, core.SIGNAL('clicked()'), self.saveFile)

        self.dispersion = self.scattering = self.dmstep = None
        self.reset()
   
    def reset(self):
        """ Reset the window ui """
        
        # Reset UI
        self.mainWidget.highFreqEdit.setText("250")
        self.mainWidget.bandwidthEdit.setText(".1953125")
        self.mainWidget.tsampEdit.setText("0.00512")
        self.mainWidget.widthEdit.setText("1")
        self.mainWidget.nchansEdit.setText("31")
        self.mainWidget.smearEdit.setText("100")
        self.mainWidget.snrEdit.setText("95")
        self.mainWidget.scatteringEdit.setText("500")

        self.mainWidget.tableWidget.setCellWidget(0, 0, gui.QLabel(" Total Dispersion"))
        self.mainWidget.tableWidget.setCellWidget(1, 0, gui.QLabel(" Subband Dispersion"))
        self.mainWidget.tableWidget.setCellWidget(2, 0, gui.QLabel(" Maximum DM"))
        self.mainWidget.tableWidget.setCellWidget(3, 0, gui.QLabel(" DM Step"))
        self.mainWidget.tableWidget.setCellWidget(4, 0, gui.QLabel(" Number of DMs"))
        self.mainWidget.tableWidget.setCellWidget(5, 0, gui.QLabel(" Numbr of Channels"))
        self.mainWidget.tableWidget.setCellWidget(6, 0, gui.QLabel(" Sampling Rate"))

        # Remove plots and values
        [ self.mainWidget.tabWidget.removeTab(0) for i in range(self.mainWidget.tabWidget.count()) ]
        [ self.mainWidget.tableWidget.setCellWidget(i, 1, gui.QLabel("")) for i in range(7) ]

        self.mainWidget.saveButton.setEnabled(False)
        self.statusBar().showMessage("Awaiting user input")
        
    def calculate(self):
        """ Calculates MDSM parameters """

        self.statusBar().showMessage("Calculating MDSM Parameters")     

        # Remove plots
        [ self.mainWidget.tabWidget.removeTab(0) for i in range(self.mainWidget.tabWidget.count()) ]

        # Interpret parameters
        try:
            tsamp = float(self.mainWidget.tsampEdit.text()) / 1000.0
            pulseWidth = float(self.mainWidget.widthEdit.text())
            nchans = int(self.mainWidget.nchansEdit.text())
            bandwidth = float(self.mainWidget.bandwidthEdit.text()) * nchans
            highFreq = 100.0 + float(self.mainWidget.highFreqEdit.text()) * float(self.mainWidget.bandwidthEdit.text()) 
            nchanslog2 = float(log(nchans,2))
            smearing = float(self.mainWidget.smearEdit.text())
            snr = float(self.mainWidget.snrEdit.text())
            scat = float(self.mainWidget.scatteringEdit.text())
        except:
            self.statusBar().showMessage('Invalid Parameters Detected!')
            return

        # Create plot objects
        self.dispersion = DispersionPlot(highFreq, bandwidth / nchans, nchans, pulseWidth, smearing)
        self.scattering = ScatteringPlot(highFreq, bandwidth, pulseWidth, scat)
        self.dmstep = DispersionStepPlot(highFreq, bandwidth, pulseWidth, tsamp)

        # Create tab placeholders
        self.mainWidget.tabWidget.addTab(self.dispersion, "Dispersion Plot")
        self.mainWidget.tabWidget.addTab(self.scattering, "Scattering Plot")
        self.mainWidget.tabWidget.addTab(self.dmstep, "Dm Step Plot")

        # Plot and show plots
        self.dispersion.plot_dispersion()
        self.scattering.plot_scattering()
        self.dmstep.plot_dmstep()  

        # Compute Parameters
        totalDisp = self.dispersion.total_dispersion()
        channelDisp = self.dispersion.channel_dispersion()
        maxDm = max(self.scattering.calculate_dm())
        dmStep = self.dmstep.calculate_optimal_dmstep(acceptedSNR = snr)
        numDms = maxDm / dmStep
        numChanslog2 = log(nchans * channelDisp * maxDm / ((smearing / 100) * pulseWidth), 2)
        numChans = int(nchans * pow(2, round(numChanslog2 - nchanslog2)))
        sRate = (tsamp / nchans) * numChans

        self.mainWidget.tableWidget.setCellWidget(0, 1, gui.QLabel("%.4f ms" % totalDisp))
        self.mainWidget.tableWidget.setCellWidget(1, 1, gui.QLabel("%.4f ms" % channelDisp))
        self.mainWidget.tableWidget.setCellWidget(2, 1, gui.QLabel("%.4f" % maxDm))
        self.mainWidget.tableWidget.setCellWidget(3, 1, gui.QLabel("%.4f" % dmStep))
        self.mainWidget.tableWidget.setCellWidget(4, 1, gui.QLabel("%d" % ceil(numDms)))
        self.mainWidget.tableWidget.setCellWidget(5, 1, gui.QLabel("%d" % ceil(numChans)))
        self.mainWidget.tableWidget.setCellWidget(6, 1, gui.QLabel("%.4f ms" % sRate))

        # Call Presto's DDPlan.py script 
        methods = presto.calculateParams(0, maxDm, highFreq - bandwidth / 2.0, bandwidth, 
                                         numChans, 32, sRate , pulseWidth * (smearing / 100))
        
        # Add the presto result to a separate tab
        table = self.mainWidget.prestoTableWidget
        table.setColumnCount(len(methods))
        table.setRowCount(8)
        header = ["Pass %d" % i for i in range(len(methods))]
        table.setHorizontalHeaderLabels(header)
        table.setVerticalHeaderLabels(["Low DM", "High DM", "dDM", "DownSamp", "dsubDM", "#DMs", "DMs/call", "calls"])
        
        for i, method in enumerate(methods):
            table.setCellWidget(0, i, gui.QLabel(" %.4f" % method.loDM))
            table.setCellWidget(1, i, gui.QLabel(" %.4f" % method.hiDM))
            table.setCellWidget(2, i, gui.QLabel(" %.4f" % method.dDM))
            table.setCellWidget(3, i, gui.QLabel(" %d" % method.downsamp))
            table.setCellWidget(4, i, gui.QLabel(" %.4f" % method.dsubDM))
            table.setCellWidget(5, i, gui.QLabel(" %d" % method.numDMs))
            table.setCellWidget(6, i, gui.QLabel(" %d" % method.DMs_per_prepsub))
            table.setCellWidget(7, i, gui.QLabel(" %d" % method.numprepsub))
            
        self.mainWidget.saveButton.setEnabled(True)
        self.statusBar().showMessage("Calculated MDSM Parameters")
        
        # Save items in class instance
        for key, value in locals().iteritems():
            self.__dict__[key] = value
                
    def saveFile(self):
        """ Saves the MDSM xml observation file """
        
        self.statusBar().showMessage("Generating MDSM observation file")  
        root = ET.Element("observation")

        freq = ET.SubElement(root, "frequencies")
        freq.set("top", str(self.highFreq))
        freq.set("offset", str(-self.bandwidth / self.numChans))
    
        dm = ET.SubElement(root, "dm")
    
        channels = ET.SubElement(root, "channels")
        channels.set("number", str(self.numChans))
        channels.set("subbands", str(32))
    
        timing = ET.SubElement(root, "timing")
        timing.set("tsamp", str(self.sRate))
    
        passes = ET.SubElement(root, "passes")
        for item in self.methods:
            currpass = ET.SubElement(passes, "pass")
            ET.SubElement(currpass, "lowDm").text = str(item.loDM)
            ET.SubElement(currpass, "highDm").text = str(item.hiDM)
            ET.SubElement(currpass, "deltaDm").text = str(item.dDM)
            ET.SubElement(currpass, "downsample").text = str(item.downsamp)
            ET.SubElement(currpass, "subDm").text = str(item.dsubDM)
            ET.SubElement(currpass, "numDms").text = str(item.numDMs)
            ET.SubElement(currpass, "dmsPerCall").text = str(item.DMs_per_prepsub)
            ET.SubElement(currpass, "ncalls").text = str(item.numprepsub)
    
        # wrap it in an ElementTree instance, and save as XML
        tree = ET.ElementTree(root)
        tree.write("observation.xml")
        
        self.statusBar().showMessage("Generated and saved XML Observation File")     
     

if __name__ == "__main__":
    app = gui.QApplication(sys.argv)
    window = MainWindow("main.ui")
    sys.exit(app.exec_())

