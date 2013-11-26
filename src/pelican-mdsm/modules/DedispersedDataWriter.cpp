#include "SpectrumDataSet.h"
#include "DedispersedDataWriter.h"
#include "DedispersedTimeSeries.h"
#include "DedispersedSeries.h"

#include <QStringList>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

// Constructor
// TODO: For now we write in 32-bit format...
DedispersedDataWriter::DedispersedDataWriter(const ConfigNode& configNode )
: AbstractOutputStream(configNode)
{
    _nSubbands = configNode.getOption("subbandsPerPacket", "value", "1").toUInt();
    _nTotalSubbands = configNode.getOption("totalComplexSubbands", "value", "1").toUInt();
    _clock = configNode.getOption("clock", "value", "200").toFloat();
    _integration    = configNode.getOption("integrateTimeBins", "value", "1").toUInt();
    _nChannels = configNode.getOption("outputChannelsPerSubband", "value", "128").toUInt();
    _nPols = configNode.getOption("nRawPolarisations", "value", "2").toUInt();

    _filePrefix = configNode.getOption("file", "prefix", "MDSM_");
    _fch1       = configNode.getOption("topChannelFrequency", "value", "150").toFloat();
    _foff       = - _clock / (_nPols * _nTotalSubbands) * float(_nSubbands);
    _tsamp      =  (_nPols * _nTotalSubbands) * _nChannels * _integration / _clock / 1e6;
    QString dms = configNode.getOption("DMs", "values", "0");

    QStringList dmList = dms.split(",", QString::SkipEmptyParts);
    std::cout << "DM Values to output: ";
    foreach(QString val, dmList) {
        _dmValues.insert(_dmValues.end(), val.toFloat());
        std::cout << val.toFloat() << " ";
    }
    std::cout << std::endl;

    // Open files
    _files.resize(_dmValues.size());
    for (unsigned i = 0; i < _files.size(); i++) {
        QString path(_filePrefix + "_" + QString::number(_dmValues[i]) + ".fil");
        _files[i] = new std::ofstream;
        _files[i] -> open(path.toUtf8().data(), std::ios::out | std::ios::binary);
    }

    // Write header in all files
    for (unsigned i = 0; i < _files.size(); i++) {
        WriteString(_files[i], "HEADER_START");
        WriteInt(_files[i], "machine_id", 0);    // Ignore for now
        WriteInt(_files[i], "telescope_id", 0);  // Ignore for now
        WriteInt(_files[i], "data_type", 1);     // Channelised Data

        WriteDouble(_files[i], "fch1", _fch1);
        WriteDouble(_files[i], "foff", _foff);
        WriteInt(_files[i], "nchans", 1);         // All channel summed
        WriteDouble(_files[i], "tsamp", _tsamp);
        WriteInt(_files[i], "nbits", 32);
        WriteDouble(_files[i], "tstart", 0);      //TODO: Extract start time from first packet
        WriteInt(_files[i], "nifs", 1);   		  // We have total power
        WriteString(_files[i], "HEADER_END");
        _files[i] -> flush();
    }
}

// Destructor
DedispersedDataWriter::~DedispersedDataWriter()
{
    for (unsigned i = 0; i < _files.size(); i++)
        _files[i] -> close();
}

// ---------------------------- Header helpers --------------------------
void DedispersedDataWriter::WriteString(std::ofstream *file, QString string)
{
    int len = string.size();
    char *text = string.toUtf8().data();
    file -> write(reinterpret_cast<char *>(&len), sizeof(int));
    file -> write(reinterpret_cast<char *>(text), len);
}

void DedispersedDataWriter::WriteInt(std::ofstream *file, QString name, int value)
{
    WriteString(file, name);
    file -> write(reinterpret_cast<char *>(&value), sizeof(int));
}

void DedispersedDataWriter::WriteDouble(std::ofstream *file, QString name, double value)
{
    WriteString(file, name);
    file -> write(reinterpret_cast<char *>(&value), sizeof(double));
}

void DedispersedDataWriter::WriteLong(std::ofstream *file, QString name, long value)
{
    WriteString(file, name);
    file -> write(reinterpret_cast<char *>(&value), sizeof(long));
}

// ---------------------------- Data helpers --------------------------

// Write data blob to disk
void DedispersedDataWriter::sendStream(const QString&, const DataBlob* incoming)
{
    DedispersedTimeSeriesF32* timeData;
    DataBlob* blob = const_cast<DataBlob*>(incoming);

    if (dynamic_cast<DedispersedTimeSeriesF32*>(blob)) {
        timeData = (DedispersedTimeSeriesF32*) dynamic_cast<DedispersedTimeSeriesF32*>(blob);
        // Check if there is data to write
        if (timeData -> nDMs() == 0)
            return;

        for(unsigned i = 0; i < _dmValues.size(); i++) {

            // Find DM time series
            for(unsigned j = 0; j < timeData -> nDMs(); j++) {
                if (fabs(timeData -> samples(j) -> dmValue() - _dmValues[i])  < 0.00001) {
                    DedispersedSeries<float>* series = timeData -> samples(j);
                    _files[i] -> write(reinterpret_cast<char *>(series -> ptr()), series -> nSamples() * sizeof(float));
                    _files[i] -> flush();
                    break;

                }
            }
        }
    }
    else {
        std::cerr << "DedispersedDataWriter::send(): "
                "Only blobs can be written by the DedispersedDataWriter" << std::endl;
        return;
    }
}
