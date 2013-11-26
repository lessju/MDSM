#include "RawVoltageAdapter.h"
#include "LofarTypes.h"

/// Constructs a new RawVoltageAdapter.
RawVoltageAdapter::RawVoltageAdapter(const ConfigNode& config)
    : AbstractStreamAdapter(config)
{
    _nBits = config.getOption("sampleSize", "bits", "0").toUInt();
    _nSamples= config.getOption("samplesPerRead", "number", "1024").toUInt();
    _nSubbands = config.getOption("subbands", "number", "1").toUInt();
    _nPolarisations = config.getOption("polarisations", "number", "1").toUInt();
    _iteration = 0;
}

/**
 * @details
 * Method to deserialise a sigproc file chunk.
 *
 * @param[in] in QIODevice poiting to an open file
 */
void RawVoltageAdapter::deserialise(QIODevice* in)
{
    // Check that data is fine
    _checkData();

    unsigned sampFactor = 1;
    unsigned dataSize= sampFactor * _nSamples * _nSubbands * _nPolarisations * _nBits / 8;
    typedef std::complex<float> fComplex;
    std::vector<char> dataTemp(dataSize);

    in -> seek(_iteration * dataSize);
    unsigned amountRead = in -> read(&dataTemp[0], dataSize);

    // If chunk size is 0, return empty blob (end of file)
    if (amountRead == 0) {
        // Reached end of file
        _timeData -> resize(0, 0, 0, 0);
        return;
    }
    else if (amountRead < dataSize) {
        // Last chunk in file (ignore?)
        _timeData -> resize(0, 0, 0, 0);
        return;
    }

    //Set timing
    _timeData -> setLofarTimestamp(_iteration * _nSamples / 32);
    _timeData -> setBlockRate(_nSamples / 32);

    // Put all the samples in one time block, converting them to complex
    unsigned dataPtr = 0;
    for(unsigned c = 0; c < _nSubbands; c++)
        for(unsigned s = 0; s < _nSamples; s++)
            for(unsigned p = 0; p < _nPolarisations; p++) {
                fComplex* data = _timeData->timeSeriesData(0, c, p);
                if (_nBits == 8)
                    ; // TODO: implement
                else if (_nBits == 16)
                {
                    unsigned char real = dataTemp[dataPtr];
                    unsigned char imag = dataTemp[dataPtr+1];

                    signed char sReal = real >= 128 ? -128 + (real & 127) : real;
                    signed char sImag = imag >= 128 ? -128 + (imag & 127) : imag;
                    data[s] = std::complex<float>(sReal, sImag);
                    dataPtr += 2 * sampFactor;
                }
            }
    _iteration++;

//    for(unsigned c = 0; c < _nSubbands; c++)
//        for(unsigned s = 0; s < _nSamples; s++)
//            for(unsigned p = 0; p < _nPolarisations; p++) {
//                fComplex* data = _timeData->timeSeriesData(0, c, p);
//                std::cout << s << ": " << data[s] << std::endl;
//}
//    exit(0);
}

/// Updates and checks the size of the time stream data.
void RawVoltageAdapter::_checkData()
{
      // Check for supported sample bits.
        if (_nBits != 8  && _nBits != 16) {
            throw QString("RawVoltageAdapter: Specified number of "
                    "sample bits (%1) not supported.").arg(_nBits);
        }

        // Check the data blob passed to the adapter is allocated.
        if (!_data) {
            throw QString("RawVoltageAdapter: Cannot deserialise into an "
                          "unallocated blob!.");
        }

        // Resize the time stream data blob being read into to match the adapter
        // dimensions.
        _timeData = static_cast<TimeSeriesDataSetC32*>(_data);
        _timeData->resize(1, _nSubbands, _nPolarisations, _nSamples);
}
