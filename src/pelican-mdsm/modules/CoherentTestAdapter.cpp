#include "CoherentTestAdapter.h"
#include "LofarTypes.h"
#include <QFile>

/// Constructs a new SigprocAdapter.
CoherentTestAdapter::CoherentTestAdapter(const ConfigNode& config)
    : AbstractStreamAdapter(config)
{
    _nSamples  = config.getOption("samplesPerRead", "number", "1024").toUInt();
    _nSubbands = config.getOption("subbands", "number", "1").toUInt();
    _tsamp     = config.getOption("tsamp", "value", "0").toFloat();
    _iteration = 0;
}

// Method to deserialise a sigproc file chunk.
void CoherentTestAdapter::deserialise(QIODevice* in)
{
    // Check that data is fine
    _checkData();

    // If first time, read file header
    if (_iteration == 0) {
        _fp = fopen( ((QFile *) in) -> fileName().toUtf8().data(),  "rb");
    }

    // Store temporary data (complex data)
    float *dataTemp = (float *) malloc(_nSamples * _nSubbands * 2 * sizeof(float));
    unsigned amountRead = fread(dataTemp, sizeof(float), _nSamples * _nSubbands * 2, _fp);

    // Check chunk size
    if (amountRead < _nSamples * _nSubbands) {
        _timeData -> resize(0, 0, 0, 0);
        return;
    }

    // Set timing
    _timeData -> setLofarTimestamp(_tsamp * _iteration * _nSamples);
    _timeData -> setBlockRate(_tsamp);

    std::complex<float> *times = _timeData -> data();
    for(unsigned s = 0; s < _nSubbands; s++)
	    for(unsigned t = 0; t < _nSamples; t++)
	    {
            unsigned index = (t * _nSubbands + s) * 2;
	        times[s * _nSamples + t] = std::complex<float>(dataTemp[index], dataTemp[index + 1]);
	    }

    _iteration++;

    free(dataTemp);
}

/// Updates and checks the size of the time stream data.
void CoherentTestAdapter::_checkData()
{
    // Check the data blob passed to the adapter is allocated.
    if (!_data)
        throw QString("SigprocAdapter: Cannot deserialise into an "
                      "unallocated blob!.");

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    _timeData = static_cast<TimeSeriesDataSetC32*>(_data);
    _timeData -> resize(_nSamples, _nSubbands, 1, 1);
}
