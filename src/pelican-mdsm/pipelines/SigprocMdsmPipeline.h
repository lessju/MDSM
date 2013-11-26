#ifndef SigprocMdsmPipeline_H
#define SigprocMdsmPipeline_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "SpectrumDataSet.h"
#include "DedispersedTimeSeries.h"
#include "DedispersedDataWriter.h"
#include "MdsmModule.h"

using namespace pelican;
using namespace pelican::lofar;

class SigprocMdsmPipeline : public AbstractPipeline
{
    public:
        SigprocMdsmPipeline();
        ~SigprocMdsmPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        MdsmModule* mdsm;

        /// Local data blobs
        SpectrumDataSetStokes* stokes;
        DedispersedTimeSeriesF32* dedispersedData;

        unsigned _iteration;
};

#endif // SigprocMdsmPipeline_H
