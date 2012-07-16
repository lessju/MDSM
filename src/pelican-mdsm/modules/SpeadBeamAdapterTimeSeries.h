#ifndef SPEAD_BEAM_ADAPTER_TIME_SERIES_H
#define SPEAD_BEAM_ADAPTER_TIME_SERIES_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "TimeSeriesDataSet.h"
#include "LofarTypes.h"
#include <complex>
/**
 * @file SpeadBeamAdapterTimeSeries.h
 */

namespace pelican {

class ConfigNode;

namespace lofar {

class TimeSeriesDataSetC32;

//
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
        void adaptHeap(unsigned heap, char* buffer, TimeSeriesDataSetC32* data);

    private:
        /// Updates and checks the size of the time stream data.
        void checkData();
        QString err(const QString& message);
        Complex makeComplex(const TYPES::i16complex& z);

    private:
        TimeSeriesDataSetC32* _timeData;
        float    _samplingTime;
        unsigned _samplesPerSubband;
        unsigned _bitsPerSample;
        unsigned _subbandsPerHeap;
        unsigned _numberOfBeams;
        unsigned _heapsPerChunk;
    	unsigned _samplesPerBlock;
        unsigned _nPolarisations;
        unsigned _iteration;

        // Temporary heap placeholder
        unsigned          _heapSize;
        std::vector<char> _heapData;
};


PELICAN_DECLARE_ADAPTER(SpeadBeamAdapterTimeSeries)

} // namespace lofar
} // namespace pelican
#endif // SPEAD_BEAM_ADAPTER_TIME_SERIES_H
