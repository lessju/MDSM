#include "ui_openDialog.h"
#include "openDialogWindow.h"
#include "mainWindow.h"

#include <qwt_plot_zoomer.h>
#include <qwt_plot_curve.h>
#include <qwt_plot.h>

#include "QMessageBox"
#include "QFile"

#include "math.h"

#include <gsl/gsl_multifit.h>
#include "omp.h"

#include <iostream>

unsigned mu = 255;

using namespace std;

// Constructor
MainWindow::MainWindow() 
{
    // Set central widget
    plotWidget = new Ui::SigprocPlotter();
    plotWidget -> setupUi(this);

    _buffer = NULL;
    _x = _y = _xB = _yB = NULL;
    _integrate = 0;
    _filesize = 0;

    // Setup
    plotWidget -> verticalGroupBox -> setEnabled(false);

    // Create and connect actions
    openAct = new QAction(tr("&Open"), this);
    openAct->setShortcuts(QKeySequence::Open);
    openAct->setStatusTip(tr("Open a Sigproc file"));
    connect(openAct, SIGNAL(triggered()), this, SLOT(openFile()));

    liveAct = new QAction(tr("&Live Files"), this);
    liveAct->setStatusTip(tr("Open MDSM dumped files"));
    connect(liveAct, SIGNAL(triggered()), this, SLOT(liveFiles()));

    saveAct = new QAction(tr("&Save Buffer"), this);
    saveAct->setShortcuts(QKeySequence::Save);
    saveAct->setStatusTip(tr("Save current buffer to file"));
    connect(saveAct, SIGNAL(triggered()), this, SLOT(saveBuffer()));

    exportAct = new QAction(tr("&Export Plot"), this);
    exportAct->setStatusTip(tr("Export current plot to disk"));
    connect(exportAct, SIGNAL(triggered()), this, SLOT(exportPlot()));

    exitAct = new QAction(tr("&Quit"), this);
    exitAct->setShortcuts(QKeySequence::Quit);
    exitAct->setStatusTip(tr("Quit"));
    connect(exitAct, SIGNAL(triggered()), this, SLOT(quit()));

    // Setup menu and assign actions
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(openAct);
    fileMenu->addAction(liveAct);
    fileMenu->addAction(saveAct);
    fileMenu->addAction(exportAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);
    
    // Connect UI controls signals
    connect(plotWidget -> channelSpin, SIGNAL(valueChanged(int)), this, SLOT(plotChannel(int)));
    connect(plotWidget -> timeSlider, SIGNAL(sliderMoved(int)), this, SLOT(sliderMoved(int)));
    connect(plotWidget -> tabWidget, SIGNAL(currentChanged(int)), this, SLOT(plot()));
    connect(plotWidget -> integrationBox, SIGNAL(valueChanged(int)), SLOT(plot()));
    connect(plotWidget -> sampleSpin, SIGNAL(valueChanged(int)), SLOT(sampleSpin(int)));
    connect(plotWidget -> beamNumber, SIGNAL(valueChanged(int)), SLOT(beamNumberChanged(int)));
    connect(plotWidget -> dmSpinBox, SIGNAL(valueChanged(double)), SLOT(dmChanged(double)));
    connect(plotWidget -> plotButton, SIGNAL(clicked()), SLOT(applyFolding()));
    connect(plotWidget -> foldingSpinBox, SIGNAL(valueChanged(int)), SLOT(foldNumberChanged(int)));
    connect(plotWidget -> channelRfiBox, SIGNAL(clicked()), SLOT(applyRFI()));
    connect(plotWidget -> spectrumRfiBox, SIGNAL(clicked()), SLOT(applyRFI()));
    connect(plotWidget -> spectrumThresholdBox, SIGNAL(valueChanged(double)), SLOT(applyRFI()));
    connect(plotWidget -> channelThresholdBox, SIGNAL(valueChanged(double)), SLOT(applyRFI()));
    connect(plotWidget -> channelBlockBox, SIGNAL(valueChanged(int)), SLOT(applyRFI()));
    connect(plotWidget -> fitDegreesBox, SIGNAL(valueChanged(int)), SLOT(applyRFI()));
    connect(plotWidget->channelMaskEdit, SIGNAL(returnPressed()), SLOT(applyRFI()));

    // Initialiase channel plot
    plotWidget -> chanPlot -> setTitle("Channel Intensity plot");

    // Initialiase bandpass plot
    plotWidget -> bandPlot -> setTitle("Bandpass plot");

    // Initialise timeseries plot
    plotWidget -> timePlot -> setTitle("Time series plot");

    // Initialiase spectogram plot
    spect = new QwtPlotSpectrogram();
    plotWidget -> specPlot -> setTitle("Channel vs Time Spectogram");
    plotWidget -> specPlot -> plotLayout()->setAlignCanvasToScales(true);
    spect -> setDisplayMode(QwtPlotSpectrogram::ImageMode, true);
    QwtLinearColorMap colorMap(Qt::black, Qt::yellow);
    spect -> setColorMap(colorMap);

    // Add vertical colour bar to spectogram
    QwtScaleWidget *rightAxis =  plotWidget -> specPlot -> axisWidget(QwtPlot::yRight);
    rightAxis->setTitle("Intensity");
    rightAxis->setColorBarEnabled(true);
    plotWidget -> specPlot -> enableAxis(QwtPlot::yRight);
    plotWidget->specPlot->setAxisTitle(QwtPlot::yLeft, "Time (in Samples)");
    plotWidget->specPlot->setAxisTitle(QwtPlot::xBottom, "Frequency Channel");

    // Add panning functionality to spectogram
    QwtPlotPanner *panner = new QwtPlotPanner(plotWidget -> specPlot -> canvas());
    panner->setAxisEnabled(QwtPlot::yRight, false);
    panner->setMouseButton(Qt::MidButton);
}

// Destructor
MainWindow::~MainWindow()
{
    free(_buffer);
    free(_x);
    free(_y);
    free(_xB);
    free(_yB);
    free(_delays);
    free(_bandpassFit);
}

// Initialise plotter
void MainWindow::initialisePlotter()
{
    // Allocate memory buffers
    _buffer = (float *) malloc(_nSamples * _nChannels * sizeof(float));
    _x      = (double *) malloc(_nSamples * sizeof(double));
    _y      = (double *) malloc(_nSamples * sizeof(double));
    _xB     = (double *) malloc(_nChannels * sizeof(double));
    _yB     = (double *) malloc(_nChannels * sizeof(double));
    _delays = (int *) malloc(_nChannels * sizeof(int));

    // Initialise UI
    plotWidget -> verticalGroupBox -> setEnabled(true);
    plotWidget -> channelSpin-> setEnabled(true);
    plotWidget -> sampleSpin -> setMaximum(_totalSamples);

    // Initialise Plots
    plotWidget -> specPlot -> setAxisScale(0, 0, _nSamples, _nSamples / 5);
    plotWidget -> specPlot -> setAxisScale(2, 0, _nChannels, _nChannels / 5);
    plotWidget -> channelSpin -> setMaximum(_nChannels);

    // Add zooming functionality to spectogram
    QwtPlotZoomer* zoomer = new SpectrogramZoomer(plotWidget -> specPlot -> canvas());
    zoomer->setMousePattern(QwtEventPattern::MouseSelect2,
                            Qt::RightButton, Qt::ControlModifier);
    zoomer->setMousePattern(QwtEventPattern::MouseSelect3,
                            Qt::RightButton);

    // Add tracker to spectogram
    const QColor c(Qt::darkBlue);
    zoomer->setRubberBandPen(c);
    zoomer->setTrackerPen(c);

    // Update UI
    plotWidget -> groupBox   -> setEnabled(true);
    plotWidget -> groupBox_2 -> setEnabled(true);
    plotWidget -> groupBox_3 -> setEnabled(true);
    plotWidget -> groupBox_4 -> setEnabled(true);
    plotWidget -> channelBlockBox->setEnabled(false);
    plotWidget -> spectrumThresholdBox->setEnabled(false);
    plotWidget -> channelThresholdBox->setEnabled(false);

    // Initialise delays array
    memset(_delays, 0, _nChannels * sizeof(int));

    // Initialise folding options
    _folded = false;

    // Initialise RFI options
    _clipChannel = _clipSpectrum = false;
    _bandpassFit = (double *) malloc(_nChannels * sizeof(double));

    // Preliminary plot
    plot(true);
}

// Load files which were (or are being) dumped by MDSM
void MainWindow::liveFiles()
{
    try
    {
        OpenDialogWindow dialog;
        dialog.exec();

        if (dialog.filenames.length() == 0)
        {
            QMessageBox msgBox;
            msgBox.setWindowTitle("Invalid File");
            msgBox.setText("No input file chosen");
            msgBox.exec();
            return;
        }

        // Here we will get a list of filenames, which we have to sort
        // and split into multiple beams
        dialog.filenames.sort();

        // Open file and get file size (should be same for each file)
        FILE *fp = fopen(dialog.filenames[0].toUtf8().data(), "rb");
        fseek(fp, 0L, SEEK_END);
        unsigned filesize = ftell(fp);
        fclose(fp);

        // Create temporary buffer to store data for each file
        // Current order: beam -> channel -> sample
        float *temp = (float *) malloc(filesize * sizeof(float) / (dialog.nBits / 8));

        // Create output file for each beam in /tmp
        FILE *beam_file[dialog.nBeams];
        for (unsigned i = 0; i < dialog.nBeams; i++)
        {
            QString filename = QString("/tmp/%1.dat").arg(i);
            beam_file[i] = fopen(filename.toAscii(), "wb");
        }

        // Process all files
        for(int i = 0; i < dialog.filenames.length(); i++)
        {
            // Open and read all the file
            fp = fopen(dialog.filenames[i].toAscii(), "rb");
            unsigned total_read = read_block(fp, dialog.nBits, temp, filesize / (dialog.nBits / 8));

            std::cout << "Processing file " << dialog.filenames[i].toAscii().data() << std::endl;

            // Set number of omp threads
            omp_set_num_threads(dialog.nBeams);
            //printf("Number of cores for openmp: %d\n", sysconf( _SC_NPROCESSORS_ONLN ) / 2);

            // If the data is Mu-Law encoded, then we have to decode it first
            if (dialog.muLawEncoded && dialog.nBits == 8)
            {
                // Total power with each value packed in 8 bits
                if (dialog.hasTotalPower)
                {
                    float log_one_plus_mu = log10(1 + mu);
                    float invserse_mu = 1.0 / mu;
                    float quant_interval = 1.0 / 255.0;

                    // Start decoding data
                    #pragma omp parallel \
                        shared(total_read, temp, quant_interval, log_one_plus_mu, invserse_mu )
                    {
                        unsigned threadId = omp_get_thread_num();
                        unsigned numThreads = omp_get_num_threads();

                        for(unsigned i = 0; i < total_read / numThreads; i++)
                        {
                            unsigned index = (total_read / numThreads) * threadId + i;
                            float datum = temp[index] * quant_interval;
                            temp[index] = (powf(10.0, log_one_plus_mu * datum) - 1) * invserse_mu;
                        }
                    }
                }
                // Complex voltages with each value packed in 8 bits (4 bits per component)
                else
                {
                    float log_one_plus_mu = log10(1 + mu);
                    float invserse_mu = 1.0 / mu;
                    float quant_interval = 1.0 / 8.0;

                    // Start decoding data
                    #pragma omp parallel \
                        shared(total_read, temp, quant_interval, log_one_plus_mu, invserse_mu )
                    {
                        unsigned threadId = omp_get_thread_num();
                        unsigned numThreads = omp_get_num_threads();

                        // Start decoding data
                        for(unsigned i = 0; i < total_read / numThreads; i++)
                        {
                            // Each char contains 2 values: real and complex
                            // containg 4 bits, one for sign and the others for the value
                            unsigned char value = temp[(total_read / numThreads) * threadId + i];

                            char real_sign  = ((value & 0x80) == 0) ? 1 : -1;
                            char imag_sign  = ((value & 0x08) == 0) ? 1 : -1;
                            char real_value = (value & 0x70) >> 4;
                            char imag_value = value & 0x07;

                            float real_datum = real_value * quant_interval;
                            float imag_datum = imag_value * quant_interval;
                            float x = (powf(10.0, (float) log_one_plus_mu * real_datum) - 1) * invserse_mu * real_sign;
                            float y = (powf(10.0, (float) log_one_plus_mu * imag_datum) - 1) * invserse_mu * imag_sign;
                            temp[(total_read / numThreads) * threadId + i] = x * x + y * y ;
                        }
                    }
                }
            }
            // If data contains voltage power, convert to total power
            else if (!dialog.hasTotalPower)
            {
                short *shortBuffer = (short *) temp;
                for(unsigned i = 0; i < total_read; i++)
                {
                    short x = shortBuffer[i * 2];
                    short y = shortBuffer[i * 2 + 1];
                    temp[i] = x * x + y * y;
                }
            }
            fclose(fp);

            unsigned nsamp = total_read / ((float) dialog.nBeams * dialog.nChannels);
            unsigned nchans = dialog.nChannels;

            // Loop over all beams and write data to files
            #pragma omp parallel
            {
                unsigned j = omp_get_thread_num();

                // Current processing beam j, loop over all samples
                for(unsigned k = 0; k < nsamp; k++)
                    // Loop over all channels
                    for(unsigned l = 0; l < nchans; l++)
                        fwrite((void *) &temp[j * nsamp * nchans + l * nsamp + k], sizeof(float), 1, beam_file[j]);
            }
        }

        // We have processed all the files! Select which beam to display
        // and initialise plotter

        // Update UI
        plotWidget -> beamNumber -> setMaximum(dialog.nBeams);

        // Get valid filename and read header
        filename = QString("/tmp/0.dat");
        file = fopen(filename.toUtf8().data(), "rb");
        _headerSize = 0;

        // Do further initialisastion
        _nBits = dialog.nBits;
        _nBeams = dialog.nBeams;
        _nSamples = dialog.nSamples;
        _nChannels = dialog.nChannels;
        _topFrequency = dialog.topFrequency;
        _bandwidth = dialog.bandwidth;
        _timestep = dialog.samplingTime;
        _hasTotalPower = dialog.hasTotalPower;

        // Get filesize
        fseek(file, 0L, SEEK_END);
        _filesize = ftell(file) - _headerSize;
        _totalSamples = _filesize / (_nChannels * (_nBits / 8));
        _currentSample = 0;
        fseek(file, 0L, SEEK_SET);

        // Change encoding parameters (since we have decoded the data already)
        _muLawEncoded = false;
        _nBits = 32;

        initialisePlotter();
    }
    catch (...)
    {
        QMessageBox msgBox;
        msgBox.setWindowTitle("Invalid File Selection");
        msgBox.setText("Invalid selection please choose files which were dumped by MDSM");
        msgBox.exec();
    }
}

// Apply folding if required
void MainWindow::applyFolding()
{
    // Apply folding to data series
    if (!_folded)
    {       
        int numFolds     = plotWidget->foldingSpinBox->value();
        _profileLength   = round((plotWidget->periodSpinBox->value() * 1e-3) / _timestep);

        // Check if we have enough values in file (from current position)
        // to generate required profile
        if (_profileLength * numFolds > _totalSamples - _currentSample || plotWidget->periodSpinBox->value() < 0.001)
        {
            QMessageBox msgBox;
            msgBox.setText(QString("Not enough data to generate profile with current settings."));
            msgBox.exec();
            _profileLength = 0;
            return;
        }

        // Set as folded
        _folded = true;

        // Update UI
        (plotWidget -> beamNumber)->setEnabled(false);
        (plotWidget -> dmSpinBox)->setEnabled(false);
        (plotWidget -> periodSpinBox)->setEnabled(false);
        (plotWidget -> groupBox_4) -> setEnabled(false);
        _clipChannel = _clipSpectrum = false;
        plotWidget->plotButton->setText("Toggle Plot Type (Data)");

        // Initialise profile file
        QFile::remove(QString("/tmp/profile.dat"));
        profileFilename = QString("/tmp/profile.dat");
        profileFile = fopen(profileFilename.toAscii(), "wb");

        // Allocate temporary memory space
        unsigned maxshift = _delays[_nChannels - 1];
        float *temp    = (float *) malloc((_profileLength + maxshift) * _nChannels * sizeof(float));
        float *profile = (float *) malloc(_profileLength * _nChannels * sizeof(float));
        memset(temp, 0, (_profileLength + maxshift) * _nChannels * sizeof(float));
        memset(profile, 0, _profileLength * _nChannels * sizeof(float));

        // Load data to generate profile
        for(unsigned p = 0; p < numFolds; p++)
        {
            // Seek file to start of current profile
            fseek(file, _headerSize + (unsigned long) (_currentSample + p * _profileLength) * _nChannels * _nBits / 8 , SEEK_SET);

            // Read current block
            read_block(file, _nBits, temp, (_profileLength + maxshift) * _nChannels);

            // Create data buffer whilst integrating and dedispersing
            for(unsigned i = 0; i < _profileLength; i++)
                for(unsigned j = 0; j < _nChannels; j++)
                        profile[i * _nChannels + j] += temp[i * _nChannels + _delays[j] * _nChannels + j];
        }

        // Write profile to file
        fwrite(profile, sizeof(float), _profileLength * _nChannels, profileFile);
        fclose(profileFile);

        // Finished generating file, set profile file as current one
        _origTotalSamples = _totalSamples;
        _origCurrentSample = _currentSample;
        _currentSample = 0;
        plot(true);

        // Clear temporary memory
        free(profile);
        free(temp);

    }
    // Don't apply folding to data series, reset plotter
    else
    {
        // Update UI
        (plotWidget -> beamNumber)->setEnabled(true);
        (plotWidget -> dmSpinBox)->setEnabled(true);
        (plotWidget -> periodSpinBox)->setEnabled(true);
        plotWidget->groupBox_4->setEnabled(true);
        _clipChannel = plotWidget->channelRfiBox->isChecked();
        _clipChannel = plotWidget->spectrumRfiBox->isChecked();
        plotWidget->plotButton->setText("Toggle Plot Type (Folded Profile)");

        _folded = false;

        plot(true);
    }
}

// Apply RFI clipping if required
void MainWindow::applyRFI()
{
    // Set global variables
    _clipChannel  = plotWidget -> channelRfiBox  -> isChecked();
    _clipSpectrum = plotWidget -> spectrumRfiBox -> isChecked();
    _channelBlock = plotWidget -> channelBlockBox -> value();
    _degrees      = plotWidget -> fitDegreesBox->value();

    // Check if we need to clip any channels
    QString mask = plotWidget->channelMaskEdit->text();
    QStringList maskList = mask.split(",", QString::SkipEmptyParts);
    _channelMask.clear();

    // For each comma-separated item, check if we have a range
    // specified as well
    for(int i = 0; i < maskList.count(); i++)
        if (maskList[i].contains(QString("-")))
        {
            // We are dealing with a range, process accordingly
            QStringList range = maskList[i].split("-", QString::SkipEmptyParts);
            unsigned from = range[0].toUInt(), to = range[1].toUInt();
            _channelMask.append(QPair<int, int>(from, to));
        }
        else
            _channelMask.append(QPair<int, int>(-1, maskList[i].toUInt()));

    // Update mask
    plot(false);
}

// Number of fold changed
void MainWindow::foldNumberChanged(int)
{
    // Nothing to do if we are not in folding mode
//    if (!_folded)
//        return;
//
//    // Recreate new pulse profile with updated number of folds
//    int numFolds     = plotWidget->foldingSpinBox->value();
//    _profileLength   = round((plotWidget->periodSpinBox->value() * 1e-3) / _timestep);
//
//    // Check if we have enough values in file (from current position)
//    // to generate required profile
//    if (_profileLength * numFolds > _origTotalSamples - _origCurrentSample || plotWidget->periodSpinBox->value() < 0.001)
//    {
//        QMessageBox msgBox;
//        msgBox.setText(QString("Not enough data to generate profile with current settings."));
//        msgBox.exec();
//        _profileLength = 0;
//        return;
//    }
//
//    // Initialise profile file
//    QFile::remove(QString("/tmp/profile.dat"));
//    profileFilename = QString("/tmp/profile.dat");
//    profileFile = fopen(profileFilename.toAscii(), "wb");
//
//    // Allocate temporary memory space
//    unsigned maxshift = _delays[_nChannels - 1];
//    float *temp    = (float *) malloc((_profileLength + maxshift) * _nChannels * sizeof(float));
//    float *profile = (float *) malloc(_profileLength * _nChannels * sizeof(float));
//    memset(temp, 0, (_profileLength + maxshift) * _nChannels * sizeof(float));
//    memset(profile, 0, _profileLength * _nChannels * sizeof(float));
//
//    // Initialise original file
//    FILE *orig_file = fopen(filename.toUtf8().data(), "rb");
//
//    // Load data to generate profile
//    for(unsigned p = 0; p < numFolds; p++)
//    {
//        // Seek file to start of current profile
//        fseek(file, _headerSize + (unsigned long) (_origCurrentSample + p * _profileLength) * _nChannels * _nBits / 8 , SEEK_SET);
//
//        // Read current block
//        read_block(orig_file, _nBits, temp, (_profileLength + maxshift) * _nChannels);
//
//        // Create data buffer whilst integrating and dedispersing
//        for(unsigned i = 0; i < _profileLength; i++)
//            for(unsigned j = 0; j < _nChannels; j++)
//                    profile[i * _nChannels + j] += temp[i * _nChannels + _delays[j] * _nChannels + j];
//    }
//
//    // Write profile to file
//    fwrite(profile, sizeof(float), _profileLength * _nChannels, profileFile);
//    fclose(profileFile);
//    fclose(orig_file);
//
//    // Finished generating file, set profile file as current one
//    plot(true);
//
//    // Clear temporary memory
//    free(profile);
//    free(temp);
}

// Open a data file to plot
void MainWindow::openFile()
{
    try
    {
        // Create open dialog instance and show
        OpenDialogWindow dialog;
        dialog.exec();

        if (dialog.filenames.length() == 0)
        {
            QMessageBox msgBox;
            msgBox.setWindowTitle("Invalid File");
            msgBox.setText("No input file chosen");
            msgBox.exec();
            return;
        }

        if (QFile::exists(dialog.filenames[0]))
        {
            // Get valid filename and read header
            filename = dialog.filenames[0];
            file = fopen(filename.toUtf8().data(), "rb");
            header = read_header(file);
            _headerSize = header == NULL ? 0 : header -> total_bytes;

            // Do further initialisastion
            _nBits = dialog.nBits;
            _nSamples = dialog.nSamples;
            _nChannels = dialog.nChannels;
            _topFrequency = dialog.topFrequency;
            _bandwidth = dialog.bandwidth;
            _timestep = dialog.samplingTime;
            _hasTotalPower = dialog.hasTotalPower;
            _muLawEncoded = dialog.muLawEncoded;

            // Get filesize
            fseek(file, 0L, SEEK_END);
            _filesize = ftell(file) - _headerSize;
            _totalSamples = _filesize / (_nChannels * (_nBits / 8));
            _currentSample = 0;
            fseek(file, 0L, SEEK_SET);

            _folded = false;

            // Initiliase plotter
            initialisePlotter();
        }
        else
        {
            QMessageBox msgBox;
            msgBox.setWindowTitle("Invalid File");
            msgBox.setText("Filepath provided does not exist");
            msgBox.exec();
        }
    }
    catch (...)
    {
        QMessageBox msgBox;
        msgBox.setWindowTitle("Invalid File");
        msgBox.setText("File does not contain sigproc data");
        msgBox.exec();
    }
}

// Save current buffer to file
void MainWindow::saveBuffer()
{
    if (_filesize == 0)
        return;

    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);

    if (dialog.exec()) {
        QStringList filenames = dialog.selectedFiles();
        if (filenames.length() == 0)
            return;

        QString filename = filenames[0];

        // Get file buffer to write
        fseek(file, _headerSize + (unsigned long) (_currentSample) * _nChannels * _nBits / 8 , SEEK_SET);
        float *curr_temp = (float *) malloc(_nSamples * _integrate * _nChannels * sizeof(float));
        read_block(file, _nBits, curr_temp, _nSamples * _integrate * _nChannels);

        FILE *fp = fopen(filename.toUtf8().data(), "wb");
        fwrite(curr_temp, sizeof(float),  _nSamples * _integrate * _nChannels, fp);
        fclose(fp);
        free(curr_temp);
    }
}

// Exit application
void MainWindow::quit() 
{
    close();
}

// Export plot to disk
void MainWindow::exportPlot()
{
    // Create QPixmap
    QPixmap pixmap(600, 600);
    pixmap.fill(Qt::white);

    // Create filter
    QwtPlotPrintFilter filter;
    int options = QwtPlotPrintFilter::PrintAll;
    options &= ~QwtPlotPrintFilter::PrintBackground;
    options |= QwtPlotPrintFilter::PrintFrameWithScales;
    filter.setOptions(options);

    // Print current plot to pixmap
    if (plotWidget -> tabWidget -> currentIndex() == 2)
        plotWidget -> bandPlot -> print(pixmap, filter);
    else if (plotWidget -> tabWidget -> currentIndex() == 1)
        plotWidget -> chanPlot -> print(pixmap, filter);
    else if (plotWidget -> tabWidget -> currentIndex() == 0)
        plotWidget -> specPlot -> print(pixmap, filter);
    else
        plotWidget -> timePlot -> print(pixmap, filter);

    // Show save file dialog
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);

    if (dialog.exec())
    {
        QStringList filenames = dialog.selectedFiles();
        if (filenames.length() == 0)
            return;

        QString filename = filenames[0];

        QFile file(filename);
        file.open(QIODevice::WriteOnly);
        pixmap.save(&file, "PNG");
    }
}

// Beam number changed, change file
void MainWindow::beamNumberChanged(int i)
{
    filename = QString("/tmp/%1.dat").arg(i - 1);
    file = fopen(filename.toUtf8().data(), "rb");
    _headerSize = 0;
    _currentSample = 0;

    // Open file, read header and update internal parameters
     file = fopen(filename.toUtf8().data(), "rb");
     header = read_header(file);
     _headerSize = (header == NULL) ? 0 : header -> total_bytes;

     fseek(file, 0L, SEEK_END);
     _filesize = ftell(file) - _headerSize;
     _totalSamples = _filesize / (_nChannels * (_nBits / 8));
     fseek(file, 0L, SEEK_SET);

    plot(false);
}

// DM changed
void MainWindow::dmChanged(double dm)
{
    // Generate new delays array
    for(unsigned i = 0; i < _nChannels; i++)
    {
        float F2 = _topFrequency;
        float F1 = _topFrequency - i * (_bandwidth / _nChannels);
        _delays[i] = (4148.741601 * ((1.0 / F1 / F1) - (1.0 / F2 / F2))) * dm / _timestep;
    }

    // Update plots
    plot(false);
}

// Slider has moved, adjust plots to reflect
// time in file
void MainWindow::sliderMoved(int i)
{
    // Slider range corresponds to 100% of the file
    // Calculate the file range (leaving enough space for the last buffer to be full)
    unsigned long maxT = _totalSamples - _nSamples * _integrate;

    // Calculate file position and seek to it
    _currentSample = round(i * maxT / 1000.0);

    // Set sample spinBox value
    plotWidget -> sampleSpin -> setValue(_currentSample);

    // Update plot
    plot(false);
}

// Finer sample control used
void MainWindow::sampleSpin(int i)
{
    // Calculate file position and seek to it
    _currentSample = i;

    // Update plot
    plot(false);
}

// Perform requried calculation for plotting
void MainWindow::createDataBuffer(unsigned integrate)
{
    // Reset buffers
    if (_folded)
    {
        float *temp   = (float *) malloc(_nSamples * _nChannels * integrate * sizeof(float));

        memset(temp, 0, _nSamples * _nChannels * sizeof(float));
        memset(_buffer, 0, _nSamples * _nChannels * sizeof(float));

        // Load file to buffer
        read_block(file, _nBits, temp, _nSamples * integrate * _nChannels);

        // Create data buffer whilst integrating and dedispersing
        for(unsigned i = 0; i < _nSamples; i++)
            for(unsigned j = 0; j < _nChannels; j++)
                for(unsigned k = 0; k < integrate; k++)
                    _buffer[i * _nChannels + j] += temp[i * integrate * _nChannels + k * _nChannels + j] / integrate;;

        free(temp);
    }
    else
    {
        unsigned maxshift = _delays[_nChannels - 1];
        float *temp   = (float *) malloc((_nSamples + maxshift) * integrate * _nChannels * sizeof(float));

        memset(temp, 0, _nSamples * _nChannels * sizeof(float));
        memset(_buffer, 0, _nSamples * _nChannels * sizeof(float));

        // Load file to buffer
        read_block(file, _nBits, temp, (_nSamples + maxshift) * integrate * _nChannels);

        // Create data buffer whilst integrating and dedispersing
        for(unsigned i = 0; i < _nSamples; i++)
            for(unsigned j = 0; j < _nChannels; j++)
                for(unsigned k = 0; k < integrate; k++)
                    _buffer[i * _nChannels + j] += temp[i * integrate * _nChannels + k * _nChannels + _delays[j] * _nChannels + j] / integrate;

        // Data buffer created, check if we need to perform any clipping (only performed on non-folded data)
        if (_clipChannel || _clipSpectrum)
        {
            // Create bandpass for fitting
            memset(_xB, 0, _nChannels * sizeof(double));
            memset(_yB, 0, _nChannels * sizeof(double));
            memset(_bandpassFit, 0, _nChannels * sizeof(double));
            for(unsigned i = 0; i < _nChannels; i++)
                _xB[i] = i / (1.0 * _nChannels);

            for(unsigned j = 0; j < _nSamples; j++)
                for(unsigned i = 0; i < _nChannels; i++)
                    _yB[i] += _buffer[j * _nChannels + i] / _nChannels;

            // Now that we have a bandpass, check if we need to mask any channels
            for(int i = 0; i < _channelMask.size(); i++)
            {
                // Check if current mask refers to a range
                if (_channelMask[i].first == -1)
                {
                    int index = _channelMask[i].second;
                    if (index == 0)
                        _yB[0] = (_yB[1] + _yB[2]) / 2.0;
                    else if (index == _nChannels - 1)
                        _yB[_nChannels - 1] = (_yB[_nChannels - 2] + _yB[_nChannels - 3]) / 2.0;
                    else
                        _yB[index] = (_yB[index-1] + _yB[index+1]) / 2.0;
                }
                else
                {
                    // Dealing with a frequency range, need to interpolate from range borders
                    float value = (_yB[_channelMask[i].first - 1], _yB[_channelMask[i].first + 1]) / 2.0;
                    for(int j = _channelMask[i].first; j <= _channelMask[i].second; j++)
                        _yB[j] = value;
                }
            }

            // Fit bandpass
            gsl_multifit_linear_workspace *ws;
            gsl_matrix *cov, *X;
            gsl_vector *y, *c;
            double chisq;

            X = gsl_matrix_alloc(_nChannels, _degrees);
            y = gsl_vector_alloc(_nChannels);
            c = gsl_vector_alloc(_degrees);
            cov = gsl_matrix_alloc(_degrees, _degrees);

            for(unsigned i = 0; i < _nChannels; i++)
            {
                gsl_matrix_set(X, i, 0, 1.0);
                for(unsigned j = 0; j < _degrees; j++)
                    gsl_matrix_set(X, i, j, pow(_xB[i], j));
                 gsl_vector_set(y, i, _yB[i]);
            }

            ws = gsl_multifit_linear_alloc(_nChannels, _degrees);
            gsl_multifit_linear(X, y, c, cov, &chisq, ws);

            // Store coefficients
            double coeffs[_nChannels];
            for(unsigned i = 0; i < _degrees; i++)
                coeffs[i] = gsl_vector_get(c, i);

            // Generate fitted bandpass
            for(unsigned i = 0; i < _nChannels; i++)
                for(unsigned j = 0; j < _degrees; j++)
                    _bandpassFit[i] += coeffs[j] * pow(_xB[i], j);

            // Calculate MSE between fit and banpass
            float mse = 0;
            for(unsigned i = 0; i < _nChannels; i++)
                mse += pow(_yB[i] - _bandpassFit[i], 2);
            mse /= _nChannels;
            mse = sqrt(mse);

            // Calculate bandpass mean and std
            float bandpass_mean = 0, bandpass_std = 0;
            for(unsigned i = 0; i < _nChannels; i++)
                 bandpass_mean += _yB[i];
            bandpass_mean /= _nChannels;

            for(unsigned i = 0; i < _nChannels; i++)
                bandpass_std += (_yB[i] - bandpass_mean) * (_yB[i] - bandpass_mean);
            bandpass_std = sqrt(bandpass_std / _nChannels);

            // Perform channel clipping if required
            if (_clipChannel)
            {
                float thresh = plotWidget->channelThresholdBox->value() * mse;

                if (_channelBlock == 1)
                {
                    for(unsigned i = 0; i < _nSamples; i++)
                        for(unsigned j = 0; j < _nChannels; j++)
                            if (_buffer[i * _nChannels + j] > thresh)
                                _buffer[i * _nChannels + j] = _bandpassFit[i];
                }
                else
                {
                    // Loop over all channels
                    for(unsigned i = 0; i < _nChannels; i++)
                    {
                        // Generate a bool array to store block rfi switch
                        unsigned blocks = _nSamples / _channelBlock;
                        bool rfi[blocks];
                        for(unsigned b = 0; b < blocks; b++) rfi[b] = false;

                        // Loop over blocks
                        for(unsigned j = 0; j < blocks; j++)
                        {
                            float value = 0;
                            for(unsigned k = 0; k < _channelBlock; k++)
                                value += _buffer[(j * _channelBlock + k) * _nChannels + i] / _channelBlock;

                            if (value > thresh)
                            {
                                rfi[j] = true;
                                if (j > 0) rfi[j-1] = true;
                                if (j < blocks - 1) rfi[j+1] = true;
                            }
                        }

                        // Mask rfi of affected blocks
                        for(unsigned j = 0; j < blocks; j++)
                            if (rfi[j])
                                for(unsigned k = 0; k < _channelBlock; k++)
                                    _buffer[(j * _channelBlock + k) * _nChannels + i] = _yB[i];
                    }
                }
            }

            // Perform spectrum clipping if required
            if (_clipSpectrum)
            {
                float thresh = bandpass_mean + plotWidget->spectrumThresholdBox->value() * bandpass_std;
                for(unsigned i = 0; i < _nSamples; i++)
                {
                    float value = 0;
                    for(unsigned j = 0; j < _nChannels; j++)
                        value += _buffer[i * _nChannels + j] / _nChannels;

                    if (value > thresh)
                        for(unsigned j = 0; j < _nChannels; j++)
                            _buffer[i * _nChannels + j] = _yB[j];
                }
            }

            gsl_multifit_linear_free(ws);
            gsl_matrix_free(X);
            gsl_matrix_free(cov);
            gsl_vector_free(y);
            gsl_vector_free(c);
        }

        free(temp);
    }
}

// Plot channel when channel number changes
void MainWindow::plotChannel(int i)
{
    // Clear memory
    memset(_x, 0, _nSamples * sizeof(double));
    memset(_y, 0, _nSamples * sizeof(double));

    // Populate buffers
    for(unsigned j = 0; j < _nSamples; j++) {
        _x[j] = j;
        _y[j] = _buffer[j * _nChannels + i];
    }

    // Update plot
    plotWidget -> chanPlot -> clear();
    QwtPlotCurve *chan = new QwtPlotCurve("Channel Intensity");
    chan -> setPen(QPen(Qt::yellow, 1));
    chan -> attach(plotWidget -> chanPlot);
    chan -> setData(_x, _y, _nSamples);
    plotWidget -> chanPlot -> replot();
}

// Update plots
void MainWindow::plot(bool reset)
{
//     try
//     {
        // Read plot parameters
        unsigned integrate = plotWidget -> integrationBox -> value();
        unsigned channel = plotWidget->channelSpin->value();
        integrate = integrate == 0 ? 1 : integrate;

        // Update integrate if required
        if (_integrate != integrate)
            _integrate = integrate;

       // Reset data file (changed integrate factor, opened new file)
       if (reset)
       {
           // Open file, read header and update internal parameters
           if (_folded)
                file = fopen(profileFilename.toUtf8().data(), "rb");
           else
                file = fopen(filename.toUtf8().data(), "rb");

            header = read_header(file);
            _headerSize = (header == NULL) ? 0 : header -> total_bytes;

            fseek(file, 0L, SEEK_END);
            _filesize = ftell(file) - _headerSize;
            _totalSamples = _filesize / (_nChannels * (_nBits / 8));
            fseek(file, 0L, SEEK_SET);

            // Check whether there is enough data for one plot
            if (_totalSamples < _nSamples * integrate)
            {
                printf("Not enough data in file...\n");
                exit(-1);
            }

            // Reset UI
            plotWidget->timeSlider->setValue(0);
            plotWidget->sampleSpin->setValue(0);
        }
       else
           // We either stay where we were (changed integrate) or the slider
           // has moved (slider guarantees that we will never go beyond the
           // end of the file
           fseek(file, _headerSize + (unsigned long) (_currentSample) * _nChannels * _nBits / 8 , SEEK_SET);

        // Update current time label
        QString time = QString("%1 s").arg(_currentSample * _timestep);
        plotWidget -> current_time_label -> setText(time.toAscii());

        // Create data buffer
        createDataBuffer(integrate);

        // ------------ Do the channel plot ------------------------
        if (plotWidget -> tabWidget -> currentIndex() == 1)
        {
            memset(_x, 0, _nSamples * sizeof(double));
            memset(_y, 0, _nSamples * sizeof(double));
            for(unsigned i = 0; i < _nSamples; i++)
            {
                _x[i] = i;
                _y[i] = _buffer[i * _nChannels + channel];
            }

            plotWidget -> chanPlot -> clear();
            plotWidget -> chanPlot -> setCanvasBackground(Qt::black);
            QwtPlotCurve *chan = new QwtPlotCurve("Channel Intensity");
            chan -> setPen(QPen(Qt::yellow, 1));
            chan -> attach(plotWidget -> chanPlot);
            chan -> setData(_x, _y, _nSamples);
            plotWidget -> chanPlot -> replot();
        }

        // ------------ Do the time series plot ------------------------
        if (plotWidget -> tabWidget -> currentIndex() == 3)
        {
            memset(_x, 0, _nSamples * sizeof(double));
            memset(_y, 0, _nSamples * sizeof(double));
            for(unsigned i = 0; i < _nSamples; i++)
            {
                _x[i] = i;
                for(unsigned j = 0; j < _nChannels; j++)
                    _y[i] += _buffer[i * _nChannels + j];
            }

            plotWidget -> timePlot -> clear();
            plotWidget -> timePlot -> setCanvasBackground(Qt::black);
            QwtPlotCurve *chan = new QwtPlotCurve("Time Series Plot");
            chan -> setPen(QPen(Qt::yellow, 1));
            chan -> attach(plotWidget -> timePlot);
            chan -> setData(_x, _y, _nSamples);
            plotWidget -> timePlot -> replot();
        }

        // ------------ Do the bandpass plot ------------------------
        if (plotWidget -> tabWidget -> currentIndex() == 2)
        {
            memset(_xB, 0, _nChannels * sizeof(double));
            memset(_yB, 0, _nChannels * sizeof(double));
            for(unsigned i = 0; i < _nChannels; i++)
                _xB[i] = i;

            for(unsigned j = 0; j < _nSamples; j++)
                for(unsigned i = 0; i < _nChannels; i++)
                {
                    float value = _buffer[j * _nChannels + i] / _nChannels;
                    _yB[i] += (value != 0) ? value : 10e-4;
                }

            plotWidget -> bandPlot -> clear();
            plotWidget -> bandPlot -> setCanvasBackground(Qt::black);

            // Add bandpass plot
            QwtPlotCurve *band = new QwtPlotCurve("Bandpass Plot");
            band -> setPen(QPen(Qt::yellow));
            band -> attach(plotWidget -> bandPlot);
            band -> setData(_xB, _yB, _nChannels);

            // If performing any clipping, add the bandpass fit as well
            if (_clipChannel || _clipSpectrum)
            {
                band = new QwtPlotCurve("Bandpass Fit Plot");
                QPen pen;
                pen.setColor(Qt::red);
                pen.setWidth(3);
                pen.setStyle(Qt::DashLine);
                band -> setPen(pen);
                band -> attach(plotWidget -> bandPlot);
                band -> setData(_xB, _bandpassFit, _nChannels);
            }
            plotWidget -> bandPlot -> replot();
        }

        // ------------ Do the spectrogram ------------------------
        if (plotWidget -> tabWidget -> currentIndex() == 0)
        {
            data = new SigprocSpectrogramData(_buffer, _nSamples-1, _nChannels-1);
            spect -> setData(*data);
            spect -> attach(plotWidget -> specPlot);

            QwtScaleWidget *rightAxis =  plotWidget -> specPlot -> axisWidget(QwtPlot::yRight);
            rightAxis->setColorMap(spect -> data().range(), spect -> colorMap());
            plotWidget -> specPlot -> setAxisScale(QwtPlot::yRight,
                                                   spect->data().range().minValue(),
                                                   spect->data().range().maxValue() );        
            plotWidget -> specPlot -> replot();
        }

//    } catch ( ... ) {
//        QMessageBox msgBox;
//        msgBox.setText("Not enough data in file");
//        msgBox.exec();
//    }
}
