#include "SigprocAdapter.h"
#include "LofarTypes.h"
#include <QFile>

/// Constructs a new SigprocAdapter.
SigprocAdapter::SigprocAdapter(const ConfigNode& config)
    : AbstractStreamAdapter(config)
{
    _nBits = config.getOption("sampleSize", "bits", "0").toUInt();
    _nSamples= config.getOption("samplesPerRead", "number", "1024").toUInt();
    _nSubbands = config.getOption("subbands", "number", "1").toUInt();
    _iteration = 0;
}

/**
 * @details
 * Method to deserialise a sigproc file chunk.
 *
 * @param[in] in QIODevice poiting to an open file
 */
void SigprocAdapter::deserialise(QIODevice* in)
{
    // Check that data is fine
    _checkData();

    // If first time, read file header
    if (_iteration == 0) {
        _fp = fopen( ((QFile *) in) -> fileName().toUtf8().data(),  "rb");
        _header = read_header(_fp);
    }

    float *dataTemp = (float *) malloc(_nSamples * _nSubbands * _nBits / 8 * sizeof(float));
    unsigned amountRead = read_block(_fp, _nBits, dataTemp, _nSamples * _nSubbands);

    // If chunk size is 0, return empty blob (end of file)
    if (amountRead == 0) {
        // Reached end of file
        _stokesData -> resize(0, 0, 0, 0);
        return;
    }
    else if (amountRead < _nSamples * _nSubbands) {
        // Last chunk in file (ignore?)
        _stokesData -> resize(0, 0, 0, 0);
        return;
    }

    //Set timing
    _stokesData -> setLofarTimestamp(_iteration * _nSamples);
    _stokesData -> setBlockRate(_nSamples);

    // Put all the samples in one time block, converting them to complex
    unsigned dataPtr = 0;
    for(unsigned s = 0; s < _nSamples; s++)
        for(unsigned c = 0; c < _nSubbands; c++) {
            float* data = _stokesData -> spectrumData(s, c, 0);
                data[0] = dataTemp[dataPtr];
                dataPtr++;
        }

    _iteration++;

    free(dataTemp);
}

/// Updates and checks the size of the time stream data.
void SigprocAdapter::_checkData()
{
    // Check for supported sample bits.
    if (_nBits != 8  && _nBits != 16 && _nBits != 32) {
        throw QString("SigprocAdapter: Specified number of "
                "sample bits (%1) not supported.").arg(_nBits);
    }

    // Check the data blob passed to the adapter is allocated.
    if (!_data) {
        throw QString("SigprocAdapter: Cannot deserialise into an "
                      "unallocated blob!.");
    }

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    _stokesData = static_cast<SpectrumDataSetStokes*>(_data);
    _stokesData->resize(_nSamples, _nSubbands, 1, 1); // 1 Channel per subband in this case (and only total power)
}
