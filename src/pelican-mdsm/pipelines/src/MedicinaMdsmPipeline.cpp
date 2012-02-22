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
    ppfChanneliser = (PPFChanneliser *) createModule("PPFChanneliser");
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

//    for (unsigned t = 0; t < 4; ++t) 
//    {
//        for(unsigned s = 0; s < 1024; ++s) 
//        {
//            Complex *data = timeSeriesData -> timeSeriesData(t, s, 0);
//            for(unsigned c = 0; c < 128; ++c) 
//            {   
//                float x = data[c].real();
//                fwrite(&x, 4, 1, fp);
//                x = data[c].imag();
//                fwrite(&x, 4, 1, fp);   
//            }
//        }
//    }

    // Run modules
    ppfChanneliser -> run(timeSeriesData, spectra);
    stokesGenerator -> run(spectra, stokes);
    mdsm -> run(stokes, dedispersedData);

    // Output dedispersed data
    dataOutput(dedispersedData, "DedispersedTimeSeriesF32");

    if (_iteration++ % 100 == 99)
        std::cout << "Processed 100 iterations" << std::endl;
}
