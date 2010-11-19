#ifndef AtaMdsmPipeline_H
#define AtaMdsmPipeline_H

#include "pelican/core/AbstractPipeline.h"
#include "SigprocStokesWriter.h"
#include "pelican/data/DataBlob.h"
#include "SpectrumDataSet.h"
#include "DedispersedTimeSeries.h"
#include "MdsmModule.h"
#include "RFI_Clipper.h"
#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "TimeSeriesDataSet.h"

using namespace pelican;
using namespace pelican::lofar;

class AtaMdsmPipeline : public AbstractPipeline
{
    public:
        AtaMdsmPipeline();
        ~AtaMdsmPipeline();

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
        SpectrumDataSetStokes* stokes;
        DedispersedTimeSeriesF32* dedispersedData;
        SpectrumDataSetC32* spectra;
        TimeSeriesDataSetC32* timeSeries;

        unsigned _iteration;
};

#endif // AtaMdsmPipeline_H
