#include "MedicinaMdsmPipeline.h"
#include <iostream>

MedicinaMdsmPipeline::MedicinaMdsmPipeline()
    : AbstractPipeline()
{
    _iteration = 0;
}

MedicinaMdsmPipeline::~MedicinaMdsmPipeline()
{ }

// Initialise the pipeline
void MedicinaMdsmPipeline::init()
{
    // Create modules
    mdsm = (MdsmModule *) createModule("MdsmModule");
//    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
    stokesGenerator = (StokesGenerator *) createModule("StokesGenerator");

    // Create local datablobs
    spectra = (SpectrumDataSetC32*) createBlob("SpectrumDataSetC32");
    stokes = (SpectrumDataSetStokes*) createBlob("SpectrumDataSetStokes");
    dedispersedData = (DedispersedTimeSeriesF32*) createBlob("DedispersedTimeSeriesF32");

    // Request remote data
    requestRemoteData("TimeSeriesDataSetC32");

    fp = fopen("output.dat", "wb");
}

// Run the pipeline
void MedicinaMdsmPipeline::run(QHash<QString, DataBlob*>& remoteData)
{
    // Get pointer to the remote TimeStreamData data blob
    timeSeriesData = (TimeSeriesDataSetC32*) remoteData["TimeSeriesDataSetC32"];

    unsigned nSamples = timeSeriesData->nTimeBlocks();
    unsigned nSubbands = timeSeriesData->nSubbands();
    unsigned nSamps = timeSeriesData->nTimesPerBlock();

    float d[nSamples * nSubbands];
    for(unsigned t = 0; t < nSamples; t++)
    for(unsigned s = 0; s < nSubbands; ++s) 
    {
        Complex *data = timeSeriesData -> timeSeriesData(t, s, 0);
        d[t*nSubbands+s] = (float) (data[0].imag() * data[0].imag()+ data[0].real() * data[0].real()); 
    }
    fwrite(d, 4, nSubbands * nSamples, fp);    
   
    // Run modules
//    ppfChanneliser -> run(timeSeriesData, spectra);
    stokesGenerator -> run(timeSeriesData, stokes);
    mdsm -> run(stokes, dedispersedData);

    // Output dedispersed data
//    dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    if (_iteration++ % 1000 == 999)
        std::cout << "Processed 1000 iterations" << std::endl;
}
