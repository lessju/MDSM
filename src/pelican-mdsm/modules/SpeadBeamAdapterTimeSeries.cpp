#include "SpeadBeamAdapterTimeSeries.h"

#include "pelican/utility/ConfigNode.h"
#include "pelican/core/AbstractStreamAdapter.h"

#include <QtCore/QString>

#include <boost/cstdint.hpp>
#include <cmath>
#include <iostream>
#include <complex>
#include <vector>

using namespace std;

namespace pelican {
namespace lofar {

// Class constructor
SpeadBeamAdapterTimeSeries::SpeadBeamAdapterTimeSeries(const ConfigNode& config)
:AbstractStreamAdapter(config), _iteration(0)
{
    if (config.type() != "SpeadBeamAdapterTimeSeries")
        throw QString("SpeadBeamAdapterTimeSeries::SpeadBeamAdapterTimeSeries(): Invalid configuration");

    // Grab configuration for the adapter
    _spectraPerSubband = config.getOption("samplesPerSubband", "value").toInt();
    _subbandsPerHeap   = config.getOption("subbandsPerHeap", "value").toInt();
    _nBeams            = config.getOption("numberOfBeams", "value", "1").toInt();
    _bitsPerSample     = config.getOption("bitsPerSample", "value").toInt();
    _nPolarisations    = config.getOption("numberOfPolarisations", "value", "1").toInt();
    _samplingTime      = config.getOption("samplingTime", "value", "0").toFloat();

    // Resize temporary heap placeholder
    _heapSize = _spectraPerSubband * _subbandsPerHeap * 
                _nBeams * _nPolarisations * _bitsPerSample / 8;
    _heapData.resize(_heapSize);
}

// Deserialise Spead heaps
void SpeadBeamAdapterTimeSeries::deserialise(QIODevice* in)
{
    // Sanity check on data blob dimensions and chunk size.
    checkData();

    // Loop over heaps
    char *heapData = &_heapData[0];

    // Loop until entire heap is read from QIODevice
    unsigned bytesRead = 0;
    int tempBytesRead = 0;
    while (bytesRead != _heapSize)
    {
        tempBytesRead = in -> read(heapData + bytesRead, _heapSize - bytesRead);
        if (tempBytesRead <= 0)  in -> waitForReadyRead(-1);
        else bytesRead += tempBytesRead;
    }

    // Interpret heap data
    adaptHeap(heapData, _timeData);

    // Set timing
    _timeData -> setTimestamp(_samplingTime * _iteration * _spectraPerSubband);
    _timeData -> setSampleRate(_samplingTime);

    _iteration++;
}

void SpeadBeamAdapterTimeSeries::adaptHeap(char* buffer, MultiBeamTimeSeriesDataSetC32* data)
{
    // NOTE: Ignores polarisation for now
    switch(_bitsPerSample)
    {
        case 32:
        {
	        i16complex *complexBuffer = reinterpret_cast<i16complex *>(buffer);
            Complex *times = data -> data();
            for(unsigned b = 0; b < _nBeams; b++)
	            for(unsigned s = 0; s < _subbandsPerHeap; s++)
		            for(unsigned t = 0; t < _spectraPerSubband; t++)
		            {
                        unsigned index = _spectraPerSubband * (b * _subbandsPerHeap + s) + t;
		                times[index] = makeComplex(complexBuffer[index]);
		            }
        }
        default:
            break;
    }
}


// Check data and module parameters
void SpeadBeamAdapterTimeSeries::checkData()
{
    // Check for supported sample bits.
    if (_bitsPerSample != 32)
        throw err("Sample size (%1 bits) not supported.").arg(_bitsPerSample);

    // Check that there is something of to adapt.
    if (_chunkSize == 0)
        cerr << "WARNING: " << err("Chunk size zero!").toStdString() << endl;

    // Check the data blob passed to the adapter is allocated.
    if (!_data) throw err("Cannot deserialise into an unallocated blob!.");

    // Check the chunk size matches the expected number of UDPPackets.
    if (_chunkSize != _nBeams * _spectraPerSubband * _subbandsPerHeap * _bitsPerSample / 8)
    {
        throw err("Chunk size '%1' != '%2.")
                   .arg(_chunkSize).arg(_nBeams * _spectraPerSubband * _subbandsPerHeap * _bitsPerSample / 8);
    }

    // Resize the time stream data blob to match the adapter dimensions.
    _timeData = (MultiBeamTimeSeriesDataSetC32*) _data;
    _timeData -> resize(_nBeams, _nPolarisations, _subbandsPerHeap, _spectraPerSubband);
}

inline QString SpeadBeamAdapterTimeSeries::err(const QString& message)
{
    return QString("SpeadBeamAdapterTimeSeries: ") + message;
}

inline SpeadBeamAdapterTimeSeries::Complex
SpeadBeamAdapterTimeSeries::makeComplex(const i16complex& z)
{
    return Complex( (Real) z.real(), (Real) z.imag() );
}

}
} // namespace pelican
