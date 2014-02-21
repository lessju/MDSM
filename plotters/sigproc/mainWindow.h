#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QFileDialog>
#include <QAction>
#include "ui_plotWidget.h"
#include "stdio.h"
#include "file_handler.h"
#include "math.h"
#include <qwt_plot_spectrogram.h>
#include <qwt_color_map.h>
#include <qwt_scale_widget.h>
#include <qwt_scale_draw.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_zoomer.h>
#include <qwt_plot_layout.h>
#include <qwt_scale_engine.h>
#include <iostream>


// SpectrogramData class for waterfall plot
class SigprocSpectrogramData: public QwtRasterData
{
    public:
        SigprocSpectrogramData(float *data, unsigned len, unsigned chans ):
            QwtRasterData(QwtDoubleRect(0, 0, chans, len))
        {  
            _len = len; _chans = chans;
            _data = data;

            _min = 9999999; _max = 0;
            for(unsigned i = 0; i < len * chans; i++) {
                if(_data[i] < _min) _min = data[i];
                if(_data[i] > _max) _max = data[i];
            }

        }

    virtual QwtRasterData *copy() const
    { return new SigprocSpectrogramData(_data, _len, _chans); }

    virtual QwtDoubleInterval range() const
    { return QwtDoubleInterval(_min, _max);  }

    virtual double value(double x, double y) const
    {
        return _data[ ((int) y ) * (_chans+1) + (int) x];
    }

    private:
        unsigned _len, _chans;
        float *_data, _min, _max;
};

class SpectrogramZoomer: public QwtPlotZoomer
{
    public:
        SpectrogramZoomer(QwtPlotCanvas *canvas):
            QwtPlotZoomer(canvas)
        { setTrackerMode(AlwaysOn); }

        virtual QwtText trackerText(const QwtDoublePoint &pos) const
        {
            QColor bg(Qt::white);
            bg.setAlpha(200);
            QwtText text = QwtPlotZoomer::trackerText(pos);
            text.setBackgroundBrush( QBrush(bg));
            return text;
        }
};

class MainWindow : public QMainWindow
{
     Q_OBJECT

    public:
        MainWindow();
        ~MainWindow();

    private slots:
        void openFile();
        void liveFiles();
        void saveBuffer();
        void quit();
        void plot(bool reset = false);
        void sliderMoved(int i);
        void dmChanged(double dm);
        void beamNumberChanged(int i);
        void plotChannel(int i);
        void sampleSpin(int i);
        void applyRFI();
        void applyFolding();
        void foldNumberChanged(int);
        void initialisePlotter();
        void exportPlot();

    private:
        void createDataBuffer(unsigned integrate);

    private:
        Ui::SigprocPlotter *plotWidget;
        FILE *file, *profileFile;
        QWidget *plotter;
        QMenu *fileMenu;
        QAction *openAct, *exitAct, *liveAct, *saveAct, *exportAct;

        FILE_HEADER* header;
        QString filename;
        QString profileFilename;

        QwtPlotSpectrogram *spect;
        SigprocSpectrogramData *data;

        unsigned _nSamples, _nChannels, _integrate, _nBeams;
        bool _hasTotalPower, _muLawEncoded;

        int _nBits;
        unsigned long _currentSample, _origTotalSamples, _origCurrentSample;
        unsigned long int _filesize, _headerSize, _totalSamples;
        float _topFrequency, _bandwidth, _timestep;

        // Folding-related variables
        int  _profileLength;
        bool _folded;

        // RFI-related variables
        bool _clipChannel, _clipSpectrum;
        double *_bandpassFit;
        int  _degrees, _channelBlock;
        QList<QPair<int, int> > _channelMask;

        // Dispersion delay array
        int *_delays;

        // Temporary memory buffers (hopefully malloced once)
        float *_buffer;
        double *_x, *_y;
        double *_xB, *_yB;
};
