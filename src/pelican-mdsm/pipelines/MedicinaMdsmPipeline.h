#ifndef MedicinaMdsmPipeline_H
#define MedicinaMdsmPipeline_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "MultiBeamTimeSeriesDataSet.h"
//#include "MdsmModule.h"

using namespace pelican;
using namespace pelican::lofar;

class MedicinaMdsmPipeline : public AbstractPipeline
{
    private:
        typedef float Real;
        typedef std::complex<Real> Complex;

    public:
        MedicinaMdsmPipeline();
        ~MedicinaMdsmPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers

        /// Local data blobs
        MedicinaStream* timeSeriesData;

        unsigned _iteration;
};

#endif // MedicinaMdsmPipeline_H
