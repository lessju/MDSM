#include "PacketChunker.h"
#include "stdio.h"
#include "stdlib.h"
#include "pthread.h"
#include "sys/time.h"

// NOTE: These are hardcoded values which depend on the backend F-engine design
// which is sending data throug 10GigE-interface using a custom packet-format
#define PACKET_LEN 256 + 8 // bytes

PacketChunker::PacketChunker(unsigned port, unsigned nAntennas, unsigned nSubbands, 
                                   unsigned nSpectra, unsigned samplesPerSecond, 
                                   unsigned packetsPerHeap)
        : _samplesPerSubband(nSpectra), _subbandsPerHeap(nSubbands), _numberOfAntennas(nAntennas), 
          _samplesPerSecond(samplesPerSecond),_packetsPerHeap(packetsPerHeap)
{   
    // Set configuration options
    _startTime = _startBlockid = 0;
    _heapSize  = nAntennas * nSubbands * nSpectra * sizeof(char);

    // Initialise chunker
    connectDevice(port);
}

PacketChunker::~PacketChunker() 
{
    // Close socket
    _socket -> close();
}

// Connect socket to start receiving data
void PacketChunker::connectDevice(unsigned port)
{
    _socket = new QUdpSocket;

    if (!_socket -> bind(port)) 
    {
        fprintf(stderr, "PacketChunker::connectDevice(): Unable to bind to UDP port!\n");
        exit(0);
    }

    // Get native socket pointer and set buffer size
    int v = 1024 * 1024 * 64;
    if (::setsockopt(_socket -> socketDescriptor(), SOL_SOCKET,
                     SO_RCVBUF, (char *) &v, sizeof(v)) == -1) 
        std::cerr << "PacketChunker::newDevice(): Unable to set socket buffer" << std::endl;
}

// Set double buffer
void PacketChunker::setDoubleBuffer(DoubleBuffer *buffer)
{
    _buffer = buffer;
    _heap = _buffer -> setHeapParameters(_subbandsPerHeap, _samplesPerSubband);
}    

// Run the UDP receiving thread
void PacketChunker::run()
{
    char     *packet;
    bool     _hasPendingPacket = 0;

    // Allocate temporary heap and packet store
    packet = (char *) malloc(PACKET_LEN);

    // Keep track of the heap being processed
    unsigned long currTime = 0;

    // Infinite reading loop
    while(true)
    {
        // Dealing with a new "heap", reset buffer
        memset(_heap, 0, _heapSize);
    
        // Reset variables
        unsigned long _currTime = 0, _numPackets = 0;
        
        // Packets per heaps is the number of packets required to get 128 spectra for all antennas/channels
        // Should be 1024 * 16
        while(_numPackets != _packetsPerHeap)
        {
            if (!_hasPendingPacket)
            {
                // Wait for next packet to be available
                while (!_socket -> hasPendingDatagrams())
                    _socket -> waitForReadyRead(1);

                // Read next packet
                if (_socket -> readDatagram(packet, PACKET_LEN) <= 0)
                    cerr << "PacketChunker: Error while receiving UDP Packet!" << std::endl;
            }
            else
                _hasPendingPacket = false;

            // We have a packet available, extract 64-bit header
            uint64_t header = ((uint64_t *) packet)[0];

            // Lower 16 bits represent antenna number (multiple by 2 to reflect the number of 
            // antennas in each packet)
            unsigned short antenna =  (header & 0x000000000000FFFF) * 2;

            // Next 10 bits represent the antenna number
            unsigned short channel = (header & 0x0000000003FF0000) >> 16;

            // Next 38 bits represent the timestamp
            unsigned long time     = (header & 0xFFFFFFFFFC000000) >> 26;

            if (_currTime == 0) _currTime = 0;

            // Check if the time in the header corresponds to the time of the
            // heap being processed, or if we've advanced one heap
            if (currTime == 0)
                currTime = time;
            else if (currTime != time)
            {
                if (time < currTime)
                {
                    std::cerr << "PacketChunker: Received out of place packer, discarding" << std::endl;
                    continue;
                }
                else
                {
                    // We are processing a packet from a new heap
                    _hasPendingPacket = true; break;
                }
            }

            // Copy packet as it is to heap buffer
            memcpy(_heap + channel * _numberOfAntennas * _samplesPerSubband + antenna * _samplesPerSubband,
                   &(packet[4]), PACKET_LEN );

            // Increment packet number
            _numPackets++;
        }


        // We have finished reading in heap
        if (_numPackets != _packetsPerHeap)
            std::cout << "Only read " << _numPackets << " for time sample " << currTime << std::endl;
        
        // Forward heap to Double Buffer (with timing parameters)
        // This will return a new heap pointer
        _heap =  _buffer -> writeHeap(1351174098.5 + (1024 * currTime) / (40e6/2.0/128.0), 1 / 19531.25);
    }
}
