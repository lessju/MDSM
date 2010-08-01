#include "MedicinaChunker.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"
#include <QtNetwork/QUdpSocket>
#include <stdio.h>
#include <iostream>
#include "pelican/utility/memCheck.h"

/**
 * @details
 * Constructs a new LofarChunker.
 *
 * TODO: this assumes variable packet size. make this a configuration option.
 */
MedicinaChunker::MedicinaChunker(const ConfigNode& config) : AbstractChunker(config)
{
    if (config.type() != "MedicinaChunker")
        throw QString("MedicinaChunker::MedicinaChunker(): Invalid configuration");

    // Get configuration options
    int _sampleType = config.getOption("samples", "type").toInt();
    _samplesPerPacket = config.getOption("params","samplesPerPacket").toInt();
    _subbandsPerPacket = config.getOption("params","subbandsPerPacket").toInt();
    _nrPolarisations = config.getOption("params","nrPolarisation").toInt();
    _clock = config.getOption("params", "clock").toInt();
    _startTime = _startBlockid = 0;
    _packetsAccepted = 0;
    _packetsRejected = 0;

    // Calculate the number of ethernet frames that will go into a chunk
    _nPackets = config.getOption("params","packets").toInt();

    // Some sanity checking.
    if (type().isEmpty())
        throw QString("MedicinaChunker::MedicinaChunker(): Data type unspecified.");

    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;

    switch (_sampleType) {
        case 4:  { _packetSize = _packetSize * sizeof(TYPES::i4complex)  + sizeof(struct UDPPacket::Header);  break; }
        case 8:  { _packetSize = _packetSize * sizeof(TYPES::i8complex)  + sizeof(struct UDPPacket::Header);  break; }
        case 16: { _packetSize = _packetSize * sizeof(TYPES::i16complex) + sizeof(struct UDPPacket::Header);  break; }
    }
}

/**
 * @details
 * Constructs a new device.
 */
QIODevice* MedicinaChunker::newDevice()
{
    QUdpSocket* socket = new QUdpSocket;
    QHostAddress hostAddress(host());
    //socket -> bind( hostAddress, port() );
    if (!socket->bind(port())) {
        std::cerr << "MedicinaChunker::newDevice(): Unabled to bind to UDP port!" << std::endl;
    }
    return socket;
}

/**
 * @details
 * Gets the next chunk of data from the UDP socket (if it exists).
 */
void MedicinaChunker::next(QIODevice* device)
{
    QUdpSocket *socket = static_cast<QUdpSocket*>(device);

    unsigned         offset                    = 0;
    unsigned         prevSeqid                 = _startTime;
    unsigned         prevBlockid               = _startBlockid;
    UDPPacket         currPacket, emptyPacket;
    qint64            sizeDatagram;

    WritableData writableData = getDataStorage( _nPackets * _packetSize);

    if (writableData.isValid()) {

        // Loop over UDP packets.
        for (int i = 0; i < _nPackets; i++) {

            // Chunker sanity check
            if (!isActive()) return;

            // Wait for datagram to be available
            while (!socket -> hasPendingDatagrams())
                socket -> waitForReadyRead(100);

            if ( ( sizeDatagram = socket -> readDatagram(reinterpret_cast<char*>(&currPacket), _packetSize) ) <= 0 ) {
                std::cout << "MedicinaChunker::next(): Error while receiving UDP Packet!" << std::endl;
                i--;
                continue;
            }

            // Check for endianness. Packet data is in little endian format
            unsigned seqid, blockid;

            #if Q_BYTE_ORDER == Q_BIG_ENDIAN
                // TODO: Convert from little endian to big endian
                seqid   = currPacket.header.timestamp;
                blockid = currPacket.header.blockSequenceNumber;
            #elif Q_BYTE_ORDER == Q_LITTLE_ENDIAN
                seqid   = currPacket.header.timestamp;
                blockid = currPacket.header.blockSequenceNumber;
            #endif

            // First time next has been run, initialise startTime and startBlockId
            if (i == 0 && _startTime == 0) {
                prevSeqid = _startTime = _startTime == 0 ? seqid : _startTime;
                prevBlockid = _startBlockid = _startBlockid == 0 ? blockid : _startBlockid;
            }

            // Sanity check in seqid. If the seconds counter is 0xFFFFFFFF,
            // the data cannot be trusted (ignore)
            if (seqid == ~0U || prevSeqid + 10 < seqid) {
                ++_packetsRejected;
                i -= 1;
                continue;
            }

            // Check that the packets are contiguous. Block id increments by no_blocks
            // which is defined in the header. Blockid is reset every interval (although
            // it might not start from 0 as the previous frame might contain data from this one)
            unsigned totBlocks = _clock * 1e6;
            unsigned lostPackets = 0, diff = 0;

/*            diff =  (blockid >= prevBlockid) ? (blockid - prevBlockid) : (blockid + totBlocks - prevBlockid);

            if (diff < _samplesPerPacket) {      // Duplicated packets... ignore
                ++_packetsRejected;
                i -= 1;
                continue;
            }
            else if (diff > _samplesPerPacket)    // Missing packets
                lostPackets = (diff / _samplesPerPacket) - 1; // -1 since it includes this includes the received packet as well

            if (lostPackets > 0) {
                printf("Generate %u empty packets, prevSeq: %u, new Seq: %u, prevBlock: %u, newBlock: %u\n",
                       lostPackets, prevSeqid, seqid, prevBlockid, blockid);
            }

            // Generate lostPackets empty packets, if any
            for (unsigned packetCounter = 0; packetCounter < lostPackets &&
                                             i + packetCounter < _nPackets; packetCounter++) {

                // Generate empty packet with correct seqid and blockid
                prevSeqid = (prevBlockid + _samplesPerPacket < totBlocks) ? prevSeqid : prevSeqid + 1;
                prevBlockid = (prevBlockid + _samplesPerPacket) % totBlocks;
                generateEmptyPacket(emptyPacket, prevSeqid, prevBlockid);
                offset = writePacket(&writableData, emptyPacket, offset);

                // Check if the number of required packets is reached
                i += 1;
                if (i == _nPackets) {
                    break;
                }
            }*/

            // Write received packet
            // FIXME: Packet will be lost if we fill up the buffer with sufficient empty packets...
            if (i != _nPackets) {
                ++_packetsAccepted;
                offset = writePacket(&writableData, currPacket, offset);
                prevSeqid = seqid;
                prevBlockid = blockid;
            }
        }
    }
    else {
        // Must discard the datagram if there is no available space.
        socket->readDatagram(0, 0);
        std::cout << "MedicinaChunker::MedicinaChunker(): "
                "Wriable data not valid, discarding packets." << std::endl;
    }

    // Update _startTime
    _startTime = prevSeqid;
    _startBlockid = prevBlockid;
    //printf("Finished receiving\n");
}

/**
 * @details
 * Generates an empty UDP packet with no time stamp.
 * TODO: packet data generation can be done once
 */
void MedicinaChunker::generateEmptyPacket(UDPPacket& packet, unsigned int seqid, unsigned int blockid)
{
    size_t size = _packetSize - sizeof(struct UDPPacket::Header);
    memset((void*) packet.data, 0, size);
    packet.header.nrBeamlets = _subbandsPerPacket;
    packet.header.nrBlocks   = _samplesPerPacket;
    packet.header.timestamp  = seqid;
    packet.header.blockSequenceNumber    = blockid;
}

/**
 * @details
 * Write packet to WritableData object
 */
int MedicinaChunker::writePacket(WritableData *writer, UDPPacket& packet, unsigned offset)
{
    if (writer -> isValid()) {
        writer -> write(reinterpret_cast<void*>(&packet), _packetSize, offset);
        return offset + _packetSize;
    } else {
        std::cerr << "MedicinaChunker::writePacket(): WritableData is not valid!" << std::endl;
        return -1;
    }
}
