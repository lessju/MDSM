#ifndef GuppiMdsmPipeline_H
#define GuppiMdsmPipeline_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "TimeSeriesDataSet.h"
#include "DedispersedTimeSeries.h"
#include "DedispersedDataWriter.h"
#include "SpectrumDataSet.h"
#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "MdsmModule.h"

using namespace pelican;
using namespace pelican::lofar;

class GuppiMdsmPipeline : public AbstractPipeline
{
    public:
        GuppiMdsmPipeline();
        ~GuppiMdsmPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        MdsmModule* mdsm;
        PPFChanneliser* ppfChanneliser;
        StokesGenerator* stokesGenerator;

        /// Local data blobs
        DedispersedTimeSeriesF32* dedispersedData;
        TimeSeriesDataSetC32* timeSeriesData;
        SpectrumDataSetStokes* intStokes;
        SpectrumDataSetStokes* stokes;
        SpectrumDataSetC32* spectra;

        unsigned _iteration;
};

#endif // GuppiMdsmPipeline_H
