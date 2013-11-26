#ifndef MDSMPIPELINE_H
#define MDSMPIPELINE_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "RFI_Clipper.h"
#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"
#include "DedispersedTimeSeries.h"
#include "MdsmModule.h"

using namespace pelican;
using namespace pelican::lofar;

class MdsmPipeline : public AbstractPipeline
{
    public:
        MdsmPipeline();
        ~MdsmPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        MdsmModule* mdsm;
        PPFChanneliser* ppfChanneliser;
        StokesGenerator* stokesGenerator;
        RFI_Clipper* rfiClipper;

        /// Local data blobs
        SpectrumDataSetC32* spectra;
        TimeSeriesDataSetC32* timeSeries;
        SpectrumDataSetStokes* stokes;
        DedispersedTimeSeriesF32* dedispersedData;
        WeightedSpectrumDataSet* weightedIntStokes;

        unsigned _iteration;
};

#endif // MDSMPIPELINE_H
