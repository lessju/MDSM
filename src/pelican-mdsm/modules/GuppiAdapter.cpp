#include "GuppiAdapter.h"
#include <complex>
#include <iostream>

#include <QFileInfo>

using namespace std;

/// Constructs a new SigprocAdapter.
GuppiAdapter::GuppiAdapter(const ConfigNode& config)
    : AbstractStreamAdapter(config)
{
    _nPolarisations = config.getOption("numberOfPolarisations", "number", "2").toUInt();
    _nSamples = config.getOption("samplesPerRead", "number", "1024").toUInt();
    _nSamplesPerTimeBlock = config.getOption("outputChannelsPerSubband", "value", "8").toUInt();
    _iteration = _dataIndex = _filesize = _processed = 0;
}

/// Read file header and extract required parameters
void GuppiAdapter::readHeader()
{
    // Check filesize

    char buf[1024 * 8];
    if ( _fp -> readLine(buf, sizeof(buf)) > 0)
    {
        QString line(buf);
        _nSubbands       = (unsigned) valueForKeyword(line, QString("OBSNCHAN"));
        _nBits           = (unsigned) valueForKeyword(line, QString("NBITS"));
        _tsamp           = valueForKeyword(line, QString("TBIN"));
        _timeSize        = (unsigned) valueForKeyword(line, QString("BLOCSIZE")) / _nSubbands;
        _dataIndex       += line.lastIndexOf("END") + 80;

        // If not set, get filesize
        if (_filesize == 0) {
            QFileInfo f(*_fp);
            _filesize = (unsigned) f.size();
        }
    }
    else
       throw QString("GuppiAdapter: File does not contain required header keywords.");
}

/// Extract the value for a given header keyword
float GuppiAdapter::valueForKeyword(QString line, QString keyword)
{
    int index = -1;
    if ( (index = line.indexOf(keyword)) == -1)
        return index;

    QString card = line.mid(index, 80);
    unsigned eqIndex = card.indexOf("=");
    QString value = card.mid(eqIndex + 1, 80 - eqIndex);
    return value.trimmed().toFloat();
}

/**
 * @details
 * Method to deserialise a guppi file chunk.
 *
 * @param[in] in QIODevice poiting to an open file
 */
void GuppiAdapter::deserialise(QIODevice* in)
{
    // If first time, read file header
    if (_iteration == 0) {
        _fp = (QFile *) in;
        readHeader();
    }

    // Check data
    _checkData();

    char data[_nSubbands][_nSamples];

    // Read file data
    if (_timeSize - _processed > _nSamples) 
    {
        // Enough data in current segment
        for (unsigned s = 0; s < _nSubbands; s++) {
            _fp -> seek(_dataIndex + s * _processed);
            _fp -> read(data[s], _nSamples);
        }

        _processed += _nSamples;
    }    
    else if (_dataIndex + _timeSize * _nSubbands >= _filesize)
    {  
        // Processing last chunk, ignore (for now)
        _timeSeriesData -> resize(0, 0, 0, 0);
        return;
    }
    else
    {
        // We need to read data from two different segments
        // Read from first segment...
        for (unsigned s = 0; s < _nSubbands; s++) {
            _fp -> seek(_dataIndex + s * _processed);
            _fp -> read(data[s], _timeSize - _processed);
        }

        unsigned prevTimeSize = _timeSize;    

        // Read next segment header
        _dataIndex += _nSubbands * _timeSize;
        _fp -> seek(_dataIndex);
        readHeader();

        // Read from second segment...
        for (unsigned s = 0; s < _nSubbands; s++) {
            _fp -> seek(_dataIndex + s * _timeSize);
            _fp -> read(&data[s][prevTimeSize - _processed], _nSamples - (prevTimeSize - _processed) );
        }

        _processed = _nSamples - (prevTimeSize - _processed);
    }

    // Set timing
    _timeSeriesData -> setLofarTimestamp(_tsamp * _iteration * _nSamples);
    _timeSeriesData -> setBlockRate(_tsamp);

    // Create lookup table
    float quantLookup[4] = { 3.3358750, 1.0, -1.0, -3.3358750 };

    // Read data from file and assign to time series data blob
    for (unsigned c = 0; c < _nSubbands; c++) {

        // Each byte contain real and imaginary parts for both polarisations
        Complex *times;

        for (unsigned t = 0; t < _nSamples; t++) {
            float xr = quantLookup[(data[c][t] >> 0) & 3];
            float xi = quantLookup[(data[c][t] >> 1) & 3];
            float yr = quantLookup[(data[c][t] >> 2) & 3];
            float yi = quantLookup[(data[c][t] >> 3) & 3]; 

            // Store X polarisation
            times = _timeSeriesData -> timeSeriesData(t / _nSamplesPerTimeBlock , c, 0);
            times[t % _nSamplesPerTimeBlock] = Complex(xr, xi);

            // Store Y polarisation
            times = _timeSeriesData -> timeSeriesData(t / _nSamplesPerTimeBlock , c, 1);
            times[t % _nSamplesPerTimeBlock] = Complex(yr, yi);
        }
    }                   

    _iteration++;
}

/// Updates and checks the size of the time stream data.
void GuppiAdapter::_checkData()
{
    // Check for supported sample bits.
    if (_nBits != 2)
        throw QString("GuppiAdapter: Only 2-bit file are supported. "
                "sample bits (%1) not supported.").arg(_nBits);

    // Check for supported number of polarisations
    if (_nPolarisations != 2)
        throw QString("GuppiAdapter: Number of polarisations (%1) not supported.").arg(_nPolarisations);

    // Check the data blob passed to the adapter is allocated.
    if (!_data) {
        throw QString("GuppiAdapter: Cannot deserialise into an "
                      "unallocated blob!.");
    }

    // Resize the time stream data blob being read into to match the adapter
    // dimensions.
    _timeSeriesData = static_cast<TimeSeriesDataSetC32*>(_data);
    _timeSeriesData->resize(_nSamples / _nSamplesPerTimeBlock, _nSubbands, 
                            _nPolarisations, _nSamplesPerTimeBlock);
}
