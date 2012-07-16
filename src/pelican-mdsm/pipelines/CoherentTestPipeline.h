#ifndef CoherentTestPipeline_H
#define CoherentTestPipeline_H

#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "TimeSeriesDataSet.h"
#include "CoherentMdsmModule.h"

using namespace pelican;
using namespace pelican::lofar;

class CoherentTestPipeline : public AbstractPipeline
{
    public:
        CoherentTestPipeline();
        ~CoherentTestPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        TimeSeriesDataSetC32 *timeData;
        CoherentMdsmModule* mdsm;
        FILE *fp;

        unsigned _iteration;
};

#endif // CoherentTestPipeline_H
