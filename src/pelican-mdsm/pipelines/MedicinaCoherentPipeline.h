#ifndef MedicinaMdsmPipeline_H
#define MedicinaMdsmPipeline_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "TimeSeriesDataSet.h"


using namespace pelican;
using namespace pelican::lofar;

class MedicinaCoherentPipeline : public AbstractPipeline
{
    private:
        typedef float Real;
        typedef std::complex<Real> Complex;

    public:
        MedicinaCoherentPipeline();
        ~MedicinaCoherentPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
//        CoherentMdsmModule* mdsm;

        /// Local data blobs
        TimeSeriesDataSetC32* timeSeriesData;

        unsigned _iteration;
//        FILE *fp1;
//        FILE *fp2;
//        FILE *fp3;
};

#endif // MedicinaMdsmPipeline_H
