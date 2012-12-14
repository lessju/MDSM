#ifndef SPEAD_BEAM_ADAPTER_TIME_SERIES_H
#define SPEAD_BEAM_ADAPTER_TIME_SERIES_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "MultiBeamTimeSeriesDataSet.h"
#include <complex>
/**
 * @file SpeadBeamAdapterTimeSeries.h
 */

namespace pelican {

class ConfigNode;

namespace lofar {

typedef std::complex<unsigned short>  i16complex;

class MultiBeamTimeSeriesDataSetC32;

class SpeadBeamAdapterTimeSeries : public AbstractStreamAdapter
{
    private:
        typedef float Real;
        typedef std::complex<Real> Complex;

    public:
        /// Constructor
        SpeadBeamAdapterTimeSeries(const ConfigNode& config);

        /// Destructor
        ~SpeadBeamAdapterTimeSeries() {}

        /// Method to deserialise a LOFAR time stream data.
        void deserialise(QIODevice* in);

        /// Adapt heap into TimeSeriesDataSetC32
        void adaptHeap(char* buffer, MultiBeamTimeSeriesDataSetC32* data);

    private:
        /// Updates and checks the size of the time stream data.
        void checkData();
        QString err(const QString& message);
        Complex makeComplex(const i16complex& z);

    private:
        MultiBeamTimeSeriesDataSetC32* _timeData;
        float    _samplingTime;
        unsigned _bitsPerSample;
        unsigned _subbandsPerHeap;
        unsigned _nBeams;
    	unsigned _spectraPerSubband;
        unsigned _nPolarisations;
        unsigned _iteration;

        // Temporary heap placeholder
        unsigned          _heapSize;
        std::vector<char> _heapData;
};


PELICAN_DECLARE_ADAPTER(SpeadBeamAdapterTimeSeries)

}
} // namespace pelican
#endif // SPEAD_BEAM_ADAPTER_TIME_SERIES_H
