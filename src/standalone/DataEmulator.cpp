#include <QHostAddress>
#include <QUdpSocket>
#include <complex>
#include <UdpHeader.h>

#include "file_handler.h"

typedef std::complex<short> i16complex;

unsigned sampPerPacket = 1, subsPerPacket = 256, 
         sampSize = 16, port = 10000, nPols = 2, sampPerSecond = 78125;

int main(int argc, char *argv[])
{
    // Read parameters, first argument must be filename!
    if (argc < 2 && (fopen(argv[1], "rb")) == NULL) {
        fprintf(stderr, "First argument must be data file!\n");
        exit(0);
    }

    // Load data file
    FILE_HEADER* header;
    FILE* fp = fopen(argv[1], "rb");
    header = read_header(fp);
//    fseek(fp, 549400, SEEK_CUR);
    unsigned nbits = 16;
       
    // Connect to host
    QUdpSocket* socket = new QUdpSocket;
    socket -> connectToHost(QHostAddress("127.0.0.1"), port);

    // Allocate data buffer
    unsigned data_read, timestamp = 0, blockid = 0, packetCounter = 0;
    float* dataBuffer = (float *) malloc(subsPerPacket * sampPerPacket * sizeof(float));

    // Initialiase packet
    UDPPacket packet;
    packet.header.version    = (char) 0xAAAB;
    packet.header.nrBeamlets = subsPerPacket;
    packet.header.nrBlocks   = sampPerPacket;
    packet.header.station    = (short) 0x00EA;
    packet.header.configuration = (char) 1010;

    // Send whole file
    do {
        // Read data from file
        data_read = read_block(fp, nbits, dataBuffer, sampPerPacket * subsPerPacket);
 
        // Convert data to packet format
        if (sampSize == 16)
        {
            // For now we put the intensity in the X Real component, the rest being 0
            i16complex *s = reinterpret_cast<i16complex*>(packet.data);
            for (unsigned i = 0; i < sampPerPacket; i++) {
                for (unsigned j = 0; j < subsPerPacket; j++) {
                    unsigned index = i * subsPerPacket * nPols + j * nPols;
                    s[index] = i16complex(dataBuffer[i * subsPerPacket + j], 0); // NOTE: inverted channels
                    s[index + 1] = i16complex(0, 0);
                }
            }
        } else {
            fprintf(stderr, "Only 16-bit complex supported for now\n");
            exit(0);
        }

        // Add timestamp
        packet.header.timestamp = 1 + (blockid + sampPerPacket) / sampPerSecond;
        packet.header.blockSequenceNumber = (blockid + sampPerPacket) % sampPerSecond;
        timestamp = packet.header.timestamp;
        blockid = packet.header.blockSequenceNumber;

        // Write data to socket
        socket -> write((char *) &packet, sizeof(packet.header) + sizeof(i16complex) * nPols * subsPerPacket * sampPerPacket);
        socket -> waitForBytesWritten(100);

        packetCounter++;
        if (packetCounter % 100000 == 0)
            printf("Generated 100000 packet\n");
        usleep(0);
    
    } while(data_read != 0);
}
