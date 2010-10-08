#include "DDStream2.h"
#include "DedispersedDataWriter.h"
#include <iostream>

DDStream2::DDStream2()
    : AbstractPipeline()
{
    _iteration = 0;
}

DDStream2::~DDStream2()
{ }

// Initialise the pipeline
void DDStream2::init()
{
    // Create modules
    mdsm = (MdsmModule *) createModule("MdsmModule");
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    rfiClipper = (RFI_Clipper *) createModule("RFI_Clipper");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    dedispersedData = (DedispersedTimeSeriesF32*) createBlob("DedispersedTimeSeriesF32");
    intStokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");

    // Request remote data
    requestRemoteData("LofarTimeStream2");
}

// Run the pipeline
void DDStream2::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeries = (TimeSeriesDataSetC32*) remoteData["LofarTimeStream2"];

    if (timeSeries -> size() == 0) {
        std::cout << "Reached end of stream" << std::endl;
        for (unsigned i = 0; i < 2; i++) { // NOTE: Too dependent on MDSM's internal state
            std::cout << "Processing extra step " << i << std::endl;
            mdsm->run(stokes, dedispersedData);
            dataOutput(dedispersedData, "DedispersedTimeSeriesF32");
            stop();
        }
    }

    // Run modules
    ppfChanneliser->run(timeSeries, spectra);
    stokesGenerator->run(spectra, stokes);
    // Clips RFI and modifies blob in place                                                                      
    rfiClipper->run(stokes);

    mdsm->run(stokes, dedispersedData);

    // Output channelised data
    //    dataOutput(stokes, "DedispersedDataWriter");

    dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    _iteration++;
}
