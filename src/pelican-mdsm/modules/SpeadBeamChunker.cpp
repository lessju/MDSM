#include "SpeadBeamChunker.h"

#include <QtNetwork/QUdpSocket>

#include <cstdio>
#include <iostream>
#include <pthread.h>

using std::cerr;
using std::cout;
using std::endl;

namespace pelican {
namespace lofar {

/**
 * @details
 * Constructs a new SpeadBeamChunker.
 */
SpeadBeamChunker::SpeadBeamChunker(const ConfigNode& config) : 
    AbstractChunker(config), _currHeapNumber(0), _hasPendingPacket(false)
{
    if (config.type() != "SpeadBeamChunker")
        throw QString("SpeadBeamChunker::SpeadBeamChunker(): Invalid configuration");

    // Get configuration options
    _samplesPerSubband = config.getOption("params", "samplesPerSubband").toInt();
    _subbandsPerHeap = config.getOption("params", "subbandsPerHeap").toInt();
    _numberOfBeams = config.getOption("params", "numberOfBeams", "1").toInt();
    _packetsPerHeap = config.getOption("params", "packetsPerHeap").toInt();
    _bitsPerSample = config.getOption("params", "bitsPerSample").toInt();

    // Some sanity checking.
    if (type().isEmpty())
        throw QString("SpeadBeamChunker::SpeadBeamChunker(): Data type unspecified.");

    _packet = (char *) malloc(SPEAD_MAX_PACKET_LEN);

    // Set thread affinity
//    pthread_t thread = pthread_self();
//    cpu_set_t cpuset;
//    CPU_ZERO(&cpuset);
//    CPU_SET(0, &cpuset);
//    CPU_SET(1, &cpuset);

//    if ((pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0)
//        perror("Cannot set pthread affinity");
}


/**
 * @details
 */
QIODevice* SpeadBeamChunker::newDevice()
{
    QUdpSocket* socket = new QUdpSocket;

    if (!socket -> bind(port()))
        cerr << "SpeadBeamBuffer::newDevice(): Unable to bind to UDP port!" << endl;

    // Get native socket pointer and set buffer size
    int v = 1024 * 1024 * 16;
    if (::setsockopt(socket -> socketDescriptor(), SOL_SOCKET,
                     SO_RCVBUF, (char *) &v, sizeof(v)) == -1) {
        std::cerr << "SpeadBeamBuffer::newDevice(): Unable to set socket buffer" << std::endl;
    }

    return socket;
}


/**
 * @details
 * Gets the next chunk of data from the UDP socket (if it exists).
 */
void SpeadBeamChunker::next(QIODevice* device)
{
    printf("Requesting next chunk\n");
    QUdpSocket *socket = static_cast<QUdpSocket *>(device);
    WritableData writableData = getDataStorage(_samplesPerSubband * _subbandsPerHeap * 
                                               _numberOfBeams * _bitsPerSample / 8);

    if (writableData.isValid())
    {
        _currHeapNumber = 0;
        _numPackets = 0;
        unsigned lost = 0;

        // Loop until we have received all packets for current heap
        while (_numPackets != _packetsPerHeap)
        {   
            // If we have a pending packet from a previous receiver, process it first
            if (!_hasPendingPacket)
            {
                // Wait for next packet to be available
                while(!socket -> hasPendingDatagrams())
                {
                    socket -> waitForReadyRead(100);
                    printf("Waiting for packet\n");
                }

                // Read next packet
                if (socket -> readDatagram(_packet, SPEAD_MAX_PACKET_LEN) <= 0)
                    cerr << "SpeadBeamChunker::next(): Error while receiving UDP Packet!" << std::endl;
            }
            else
                _hasPendingPacket = false; 

            printf("Received packet\n");

            // Unpack packet header (64 bits)
            uint64_t hdr;
            hdr = SPEAD_HEADER(_packet);

            if ((SPEAD_GET_MAGIC(hdr) != SPEAD_MAGIC) ||
                    (SPEAD_GET_VERSION(hdr) != SPEAD_VERSION) ||
                    (SPEAD_GET_ITEMSIZE(hdr) != SPEAD_ITEM_PTR_WIDTH) ||
                    (SPEAD_GET_ADDRSIZE(hdr) != SPEAD_HEAP_ADDR_WIDTH))
                continue;

            unsigned nItems = SPEAD_GET_NITEMS(hdr);
            char *payload = _packet + SPEAD_HEADERLEN + nItems * SPEAD_ITEMLEN;

            // Unpack packet items: Each item is 64 bits wide and all beam items use direct mode addressing
            uint64_t heapSize, payloadOffset, payloadLen, beamId, item, heapNumber;

            // Item 1: heap number
            item = SPEAD_ITEM(_packet, 1);
            heapNumber = SPEAD_ITEM_ADDR(item);

            // Item 2: heap size
            item = SPEAD_ITEM(_packet, 2);
            heapSize = SPEAD_ITEM_ADDR(item);

            // Item 3: payload length
            item = SPEAD_ITEM(_packet, 3);
            payloadOffset = SPEAD_ITEM_ADDR(item);

            // item 4: item number
            item = SPEAD_ITEM(_packet, 4);
            payloadLen = SPEAD_ITEM_ADDR(item);

            // Item 5: data
            item = SPEAD_ITEM(_packet, 5);
            beamId = SPEAD_ITEM_ID(item) - 6000 * 8; // -6000 * 8 for beam id .... for some reason


            // Some sanity checking
            if (_currHeapNumber == 0)
                _currHeapNumber = heapNumber;
            else if (heapNumber != _currHeapNumber)
            {
                if (heapNumber < _currHeapNumber)
                    std::cout << "SpeadBeamChunker::next(): Received out of place packet, discarding" << std::endl;
                else 
                {
                    // We are processing a packet from a new heap
                    std::cout << "Processing packet from heap " << heapNumber << " current heap: " <<_currHeapNumber << std::endl;
                    _hasPendingPacket = true;
                    break;
                }
            }

            // Add packet writableData object
            unsigned offset = beamId * _samplesPerSubband * _subbandsPerHeap * _bitsPerSample / 8 + payloadOffset;
            writePacket(&writableData, payload, payloadLen, offset);

//                std::cout << _numPackets << " " << heapNumber << " " << heapSize << " =" << payloadOffset << "= " << payloadLen << " " << beamId << " " << offset << " " << offset - prev_offset << " "  << std::endl;
            _numPackets++;
        }

        if (_numPackets != _packetsPerHeap)
            std::cout << "Only read " << _numPackets << " from heap [" << _currHeapNumber << "], lost: " << lost  << std::endl;
    }
    else
    {
        socket -> readDatagram(0, 0);
        cout << "SpeadBeamChunker::next(): WritableData not valid, discarding packets" << std::endl;
    }
}

/**
 * @details
 * Write heap to WritableData object
 */
void SpeadBeamChunker::writePacket(WritableData *writer, char *data, unsigned size, unsigned offset)
{
    if (writer -> isValid()) 
        writer -> write(reinterpret_cast<void *>(data), size, offset);
    else 
        cerr << "SpeadBeamChunker::writeHeap(): WritableData is not valid!" << endl;
}


} // namespace lofar
} // namespace pelican
