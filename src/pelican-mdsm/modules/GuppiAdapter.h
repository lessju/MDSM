#ifndef GuppiAdapter_H
#define GuppiAdapter_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "TimeSeriesDataSet.h"
#include <complex>
#include <QFile>

using namespace pelican;
using namespace pelican::lofar;

class GuppiAdapter: public AbstractStreamAdapter
{
    private:

        typedef float Real;
        typedef std::complex<Real> Complex;

    public:
        /// Constructs a new SigprocAdapter.
        GuppiAdapter(const ConfigNode& config);

        /// Destroys the SigprocAdapter.
        ~GuppiAdapter() {}

    protected:
        /// Method to deserialise a sigproc file
        void deserialise(QIODevice* in);

    private:
        /// Updates and checks the size of the time stream data.
        void _checkData();

        /// Extract the value for a given header keyword
        float valueForKeyword(QString line, QString keyword);

        /// Read file header and extract required parameters
        void readHeader();

    private:
        TimeSeriesDataSetC32* _timeSeriesData;
        QFile *_fp;

        unsigned long int _iteration;
        unsigned _nSamplesPerTimeBlock;
        unsigned _nSamples;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nBits;
        double   _tsamp;

        unsigned _processed;
        unsigned _timeSize;
        unsigned _filesize;
        unsigned _dataIndex;
};

PELICAN_DECLARE_ADAPTER(GuppiAdapter)

#endif // GuppiAdapter_H
