#include "ChannelisedStreamData.h"
#include "TimeStreamData.h"
#include "SigprocWriter.h"

#include <iostream>
#include <fstream>

// Constructor
// TODO: For now we write in 32-bit format...
SigprocWriter::SigprocWriter(const ConfigNode& configNode )
      : AbstractOutputStream(configNode)
{
    // Initliase connection manager thread
    _filepath = configNode.getOption("file", "filepath");
    _fch1     = configNode.getOption("params", "channelOneFrequency", "0").toFloat();
    _foff     = configNode.getOption("params", "frequencyOffset", "0").toFloat();
    _tsamp    = configNode.getOption("params", "samplingTime", "0").toFloat();
    _nchans    = configNode.getOption("params", "numberOfChannels", "31").toFloat();

    // Open file
    _file.open(_filepath.toUtf8().data(), std::ios::out | std::ios::binary);

    // Write header
    WriteString("HEADER_START");
    WriteString("Telescope");
    WriteString("LOFAR");
    WriteInt("machine_id", 0);    // Ignore for now
    WriteInt("telescope_id", 0);  // Ignore for now
    WriteInt("data_type", 1);     // Channelised Data

    // Need to be parametrised ...
    WriteDouble("fch1", _fch1);
    WriteDouble("foff", _foff);
    WriteInt("nchans", _nchans);
    WriteDouble("tsamp", _tsamp);
    WriteInt("nbits",32);          // Only 32-bit binary data output is implemented for now
    WriteDouble("tstart", 0);      //TODO: Extract start time from first packet
    WriteInt("nifs", 1);           // We don't have intermediate frequencies
    WriteString("HEADER_END");
}

// Destructor
SigprocWriter::~SigprocWriter()
{
    _file.close();
}

// ---------------------------- Header helpers --------------------------
void SigprocWriter::WriteString(QString string)
{
    int len = string.size();
    char *text = string.toUtf8().data();
    _file.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file.write(reinterpret_cast<char *>(text), len);
}

void SigprocWriter::WriteInt(QString name, int value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(int));
}

void SigprocWriter::WriteDouble(QString name, double value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(double));
}

void SigprocWriter::WriteLong(QString name, long value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(long));
}

// ---------------------------- Data helpers --------------------------

// Write data blob to disk
void SigprocWriter::send(const QString& streamName, const DataBlob* incoming)
{
    std::complex<double> *data;
    unsigned int size, i;

    DataBlob* blob = const_cast<DataBlob *>(incoming);
    if (dynamic_cast<ChannelisedStreamData *>(blob)) {
        ChannelisedStreamData *cData = (ChannelisedStreamData *) dynamic_cast<ChannelisedStreamData *>(blob);
        data = cData -> data();  
        size = cData -> size();
    }
    else if (dynamic_cast<TimeStreamData *>(blob)) {
        TimeStreamData *tData = (TimeStreamData *) dynamic_cast<TimeStreamData *>(blob);
        data = tData -> data();  
        size = tData -> size();
    } else {
        std::cerr << "Only ChannelisedStreamData blobs can be written by the SigprocWriter" << std::endl;
        return;
    }

    float *buffer = (float *) malloc(size * sizeof(float));

    // Calculate total power in complex values
    for(i = 0; i < size; i++)
        buffer[i] = sqrt(pow(data[i].real(), 2) + pow(data[i].imag(), 2));

    // Write data to file
    _file.write(reinterpret_cast<char *> (buffer), size * sizeof(float));
    _file.flush();
}
