#ifndef MDSMPIPELINE_H
#define MDSMPIPELINE_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "TimeStreamData.h"
#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "SubbandSpectra.h"
#include "SubbandTimeSeries.h"
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
//
//        /// Local data blobs
        SubbandSpectraC32* spectra;
        SubbandTimeSeriesC32* timeSeries;
        SubbandSpectraStokes* stokes;

        unsigned _iteration;
};

#endif // MDSMPIPELINE_H 
