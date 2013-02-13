#include "ui_openDialog.h"
#include "openDialogWindow.h"
#include "mainWindow.h"

#include <qwt_plot_zoomer.h>
#include <qwt_plot_curve.h>
#include <qwt_plot.h>

#include "QMessageBox"
#include "QFile"

#include "math.h"

#include <iostream>

unsigned mu = 255;

// Destructor
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

    exitAct = new QAction(tr("&Quit"), this);
    exitAct->setShortcuts(QKeySequence::Quit);
    exitAct->setStatusTip(tr("Quit"));
    connect(exitAct, SIGNAL(triggered()), this, SLOT(quit()));

    // Setup menu and assign actions
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(openAct);
    fileMenu->addAction(liveAct);
    fileMenu->addAction(saveAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);
    
    // Connect UI controls signals
    connect(plotWidget -> plotButton, SIGNAL(clicked()), this, SLOT(plot()));   
    connect(plotWidget -> channelSpin, SIGNAL(valueChanged(int)), this, SLOT(plotChannel(int)));
    connect(plotWidget -> timeSlider, SIGNAL(sliderMoved(int)), this, SLOT(sliderMoved(int)));
    connect(plotWidget -> tabWidget, SIGNAL(currentChanged(int)), this, SLOT(plot()));
    connect(plotWidget -> integrationBox, SIGNAL(valueChanged(int)), SLOT(plot()));
    connect(plotWidget -> sampleSpin, SIGNAL(valueChanged(int)), SLOT(sampleSpin(int)));
    connect(plotWidget -> beamNumber, SIGNAL(valueChanged(int)), SLOT(beamNumberChanged(int)));

    // Initialiase channel plot
    plotWidget -> chanPlot -> setTitle("Channel Intensity plot");

    // Initialiase bandpass plot
    plotWidget -> bandPlot -> setTitle("Bandpass plot");

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
MainWindow::~MainWindow() { }

// Initialise plotter
void MainWindow::initialisePlotter()
{
    // Allocate memory buffers
    _temp   = (float *) malloc(_nSamples * _nChannels * sizeof(float));
    _buffer = (float *) malloc(_nSamples * _nChannels * sizeof(float));
    _x      = (double *) malloc(_nSamples * sizeof(double));
    _y      = (double *) malloc(_nSamples * sizeof(double));
    _xB     = (double *) malloc(_nChannels * sizeof(double));
    _yB     = (double *) malloc(_nChannels * sizeof(double));

    // Initialise UI
    plotWidget -> verticalGroupBox -> setEnabled(true);
    plotWidget -> channelSpin->setEnabled(true);
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
    plotWidget->groupBox->setEnabled(true);
    plotWidget->groupBox_2->setEnabled(true);
    plotWidget->plotButton->setEnabled(true);

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
        for(unsigned i = 0; i < dialog.filenames.length(); i++)
        {
            // Open and read all the file
            fp = fopen(dialog.filenames[i].toAscii(), "rb");
            unsigned total_read = read_block(fp, dialog.nBits, temp, filesize / (dialog.nBits / 8));

            // If the data is Mu-Law encoded, then we have to decode it first
            if (dialog.muLawEncoded && dialog.nBits == 8)
            {
                float log_one_plus_mu = log10(1 + mu);
                float invserse_mu = 1.0 / mu;
                float quant_interval = 1.0 / 255.0;

                // Start decoding data
                for(unsigned i = 0; i < total_read; i++)
                {
                    float datum = temp[i] * quant_interval;
                    temp[i] = (powf(10.0, log_one_plus_mu * datum) - 1) * invserse_mu;
                }
            }
            fclose(fp);

            unsigned nsamp = total_read / ((float) dialog.nBeams * dialog.nChannels);
            unsigned nchans = dialog.nChannels;
            unsigned nbeams = dialog.nBeams;

            // Loop over all beams and write data to files
            for(unsigned j = 0; j < nbeams; j++)
                // Current processing beam j, loop over all samples
                for(unsigned k = 0; k < nsamp; k++)
                    // Loop over all channels
                    for(unsigned l = 0; l < nchans; l++)
                        fwrite((void *) &temp[j * nsamp * nchans + l * nsamp + k], sizeof(float), 1, beam_file[j]);
        }

        // We have processed all the files! Select which beam to display
        // and initialise plotter

        // Update UI
        plotWidget -> beamNumber -> setMaximum(dialog.nBeams);

        // Get valid filename and read header
        filename = QString("/tmp/%1.dat").arg(plotWidget->beamNumber->value() - 1);
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

// Beam number changed, change file
void MainWindow::beamNumberChanged(int i)
{
    filename = QString("/tmp/%1.dat").arg(i - 1);
    file = fopen(filename.toUtf8().data(), "rb");
    _headerSize = 0;
    _currentSample = 0;

    plot(true);
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
    memset(_temp, 0, _nSamples * _nChannels * sizeof(float));
    memset(_buffer, 0, _nSamples * _nChannels * sizeof(float));

    // Check if we have enough memory space
    if (integrate > _nSamples)
    {
        fprintf(stderr, "Integration limit is %d\n", _nSamples);
        exit(0);
    }

    // Read data from file whilst integrating
    for(unsigned i = 0; i < _nSamples; i++)
    {
        read_block(file, _nBits, _temp, integrate * _nChannels);

        // If the data is Mu-Law encoded, then we have to decode it first
        if (_muLawEncoded && _nBits == 8)
        {
            float log_one_plus_mu = log10(1 + mu);
            float invserse_mu = 1.0 / mu;
            float quant_interval = 1.0 / 255.0;

            // Start decoding data
            for(unsigned i = 0; i < integrate * _nChannels; i++)
            {
                float datum = _temp[i] * quant_interval;
                _temp[i] = (powf(10.0, log_one_plus_mu * datum) - 1) * invserse_mu;
            }
        }

        for(unsigned j = 0; j < _nChannels; j++)
            for(unsigned k = 0; k < integrate; k++)
                _buffer[i * _nChannels + j] += _temp[k * _nChannels + j] / integrate;
    }
}

void MainWindow::plotChannel(int i)
{
    // Clear memory
    memset(_x, 0, _nSamples * sizeof(double));
    memset(_y, 0, _nSamples * sizeof(double));

    // Populate buffers
    unsigned channel = plotWidget->channelSpin->value();
    for(unsigned i = 0; i < _nSamples; i++) {
        _x[i] = i;
        _y[i] = _buffer[i * _nChannels + channel];
    }

    // Update plot
    plotWidget -> chanPlot -> clear();
    QwtPlotCurve *chan = new QwtPlotCurve("Channel Intensity");
    chan -> setPen(QPen(Qt::yellow, 1));
    chan -> attach(plotWidget -> chanPlot);
    chan -> setData(_x, _y, _nSamples);
    plotWidget -> chanPlot -> replot();
}

void MainWindow::plot(bool reset)
{
     try
     {
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

        // ------------ Do the bandpass plot ------------------------
        if (plotWidget -> tabWidget -> currentIndex() == 2)
        {
            memset(_xB, 0, _nChannels * sizeof(double));
            memset(_yB, 0, _nChannels * sizeof(double));
            for(unsigned i = 0; i < _nChannels; i++)
                _xB[i] = i;

            for(unsigned j = 0; j < _nSamples; j++)
                for(unsigned i = 0; i < _nChannels; i++)
                    _yB[i] += 10 * log10(_buffer[j * _nChannels + i] / _nChannels);

            plotWidget -> bandPlot -> clear();
            plotWidget -> bandPlot -> setCanvasBackground(Qt::black);
            QwtPlotCurve *band = new QwtPlotCurve("Bandpass Plot");
            band -> setPen(QPen(Qt::yellow));
            band -> attach(plotWidget -> bandPlot);
            band -> setData(_xB, _yB, _nChannels);
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

    } catch ( ... ) {
        QMessageBox msgBox;
        msgBox.setText("Not enough data in file");
        msgBox.exec();
    }
}
