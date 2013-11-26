#ifndef DD_STREAM1_H
#define DD_STREAM1_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "pelican/output/PelicanTCPBlobServer.h"

#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "RFI_Clipper.h"
#include "StokesIntegrator.h"
#include "MdsmModule.h"
#include "AdapterTimeSeriesDataSet.h"
#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"

#include "SigprocStokesWriter.h"

/**
 * @file UdpBFPipeline.h
 */

namespace pelican {
namespace lofar {

/**
 * @class UdpBFPipeline
 *
 * @brief
 *
 * @details
 *
 */
class DDStream1 : public AbstractPipeline
{
    public:
        DDStream1();
        ~DDStream1();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        MdsmModule* mdsm;
        PPFChanneliser* ppfChanneliser;
        StokesGenerator* stokesGenerator;
        StokesIntegrator* stokesIntegrator;
        RFI_Clipper* rfiClipper;

        /// Local data blob
        SpectrumDataSetC32* spectra;
        TimeSeriesDataSetC32* timeSeries;
        SpectrumDataSetStokes* stokes;
        SpectrumDataSetStokes* intStokes;
        DedispersedTimeSeriesF32* dedispersedData;
        WeightedSpectrumDataSet* weightedIntStokes;

        unsigned _iteration;
};

} // namespace lofar
} // namespace pelican

#endif // DD_STREAM1_H
