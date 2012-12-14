#include "SpeadBeamChunker.h"
#include "Types.h"
#include "stdio.h"
#include "stdlib.h"
#include "pthread.h"
#include "sys/time.h"

SpeadBeamChunker::SpeadBeamChunker(unsigned port, unsigned nBeams, unsigned nSubbands, 
                                   unsigned nSpectra, unsigned samplesPerSecond, 
                                   unsigned packetsPerHeap)
        : _samplesPerSubband(nSpectra), _subbandsPerHeap(nSubbands), _numberOfBeams(nBeams), 
          _samplesPerSecond(samplesPerSecond),_packetsPerHeap(packetsPerHeap)
{   
    // Set configuration options
    _startTime = _startBlockid = 0;
    _heapSize  = nBeams * nSubbands * nSpectra * sizeof(float);

    // Initialise chunker
    connectDevice(port);

    // Set thread affinity
//    pthread_t thread = pthread_self();
//    cpu_set_t cpuset;
//    CPU_ZERO(&cpuset);
//    CPU_SET(0, &cpuset);

//    if ((pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0)
//        perror("Cannot set pthread affinity");
}

SpeadBeamChunker::~SpeadBeamChunker() 
{
    // Close socket
    _socket -> close();
}

// Connect socket to start receiving data
void SpeadBeamChunker::connectDevice(unsigned port)
{
    _socket = new QUdpSocket;

    if (!_socket -> bind(port)) 
    {
        fprintf(stderr, "SpeadBeamChunker::connectDevice(): Unable to bind to UDP port!\n");
        exit(0);
    }

    // Get native socket pointer and set buffer size
    int v = 1024 * 1024 * 64;
    if (::setsockopt(_socket -> socketDescriptor(), SOL_SOCKET,
                     SO_RCVBUF, (char *) &v, sizeof(v)) == -1) 
        std::cerr << "SpeadBeamBuffer::newDevice(): Unable to set socket buffer" << std::endl;
}

// Set double buffer
void SpeadBeamChunker::setDoubleBuffer(DoubleBuffer *buffer)
{
    _buffer = buffer;
    _heap = _buffer -> setHeapParameters(_subbandsPerHeap, _samplesPerSubband);
}    

// Run the UDP receiving thread
void SpeadBeamChunker::run()
{
    char     *packet;
    bool     _hasPendingPacket = 0;

    // Allocate temporary heap and packet store
    packet = (char *) malloc(SPEAD_MAX_PACKET_LEN);

    // Packet items: Each item is 64 bits wide and all beam items use direct mode addressing
    uint64_t heapSize, payloadOffset, payloadLen, beamId, item, heapNumber;

    // Infinite reading loop
    while(true)
    {
        // Dealing with a new heap, reset heap data
        memset(_heap, 0, _heapSize);

	    // Reset variables
	    unsigned _currHeapNumber = 0, _numPackets = 0;
	
        // Loop until we have received all packets for current heap
        while (_numPackets != _packetsPerHeap)
        {   
            if (!_hasPendingPacket)
            {
                // Wait for next packet to be available
                while(!_socket -> hasPendingDatagrams())
                    _socket -> waitForReadyRead(1);

                // Read next packet
                if (_socket -> readDatagram(packet, SPEAD_MAX_PACKET_LEN) <= 0)
                    cerr << "SpeadBeamChunker::next(): Error while receiving UDP Packet!" << std::endl;
            }
            else
                _hasPendingPacket = false; 

            // Unpack packet header (64 bits)
            uint64_t hdr;
            hdr = SPEAD_HEADER(packet);

            if ((SPEAD_GET_MAGIC(hdr) != SPEAD_MAGIC) ||
                    (SPEAD_GET_VERSION(hdr) != SPEAD_VERSION) ||
                    (SPEAD_GET_ITEMSIZE(hdr) != SPEAD_ITEM_PTR_WIDTH) ||
                    (SPEAD_GET_ADDRSIZE(hdr) != SPEAD_HEAP_ADDR_WIDTH))
                continue;

            unsigned nItems = SPEAD_GET_NITEMS(hdr);
            char *payload = packet + SPEAD_HEADERLEN + nItems * SPEAD_ITEMLEN;

            // Item 1: heap number
            item = SPEAD_ITEM(packet, 1);
            heapNumber = SPEAD_ITEM_ADDR(item);

            // Item 2: heap size
            item = SPEAD_ITEM(packet, 2);
            heapSize = SPEAD_ITEM_ADDR(item);

            // Item 3: payload offset
            item = SPEAD_ITEM(packet, 3);
            payloadOffset = SPEAD_ITEM_ADDR(item);

            // item 4: payload length
            item = SPEAD_ITEM(packet, 4);
            payloadLen = SPEAD_ITEM_ADDR(item);

            // Item 5: beam ID
            item = SPEAD_ITEM(packet, 5);
            beamId = SPEAD_ITEM_ID(item) - 6000 * 8; // -6000 * 8 for beam id .... for some reason

            // Some sanity checking
            if (_currHeapNumber == 0)
                _currHeapNumber = heapNumber;
            else if (heapNumber != _currHeapNumber)
            {
                if (heapNumber < _currHeapNumber)
                    std::cout << "SpeadBeamChunker::next(): Received out of place packet, discarding" 
                              << std::endl;
                else // We are processing a packet from a new heap
                {  _hasPendingPacket = true; break; }
            }

            // Add packet heap data object
            memcpy(_heap + payloadOffset + heapSize * beamId, payload, payloadLen);

	        // Increment packet number
	        _numPackets++;
        }

        // We have finished reading in heap
        if (_numPackets != _packetsPerHeap)
            std::cout << "Only read " << _numPackets << " from heap [" 
                      << _currHeapNumber << "]"  << std::endl;
        
        // Forward heap to Double Buffer (with timing parameters)
        // This will return a new heap pointer
        _heap =  _buffer -> writeHeap(1351174098.5 + (1024 * heapNumber) / (40e6/2.0/128.0), 1 / 19531.25);
    }
}
