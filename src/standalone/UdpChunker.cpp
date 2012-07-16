#include "UdpChunker.h"
#include "Types.h"
#include "stdio.h"
#include "stdlib.h"

UDPChunker::UDPChunker(unsigned port, unsigned samplesPerPacket, unsigned nSubbands, 
                       unsigned nPolarisations, unsigned samplesPerSecond, unsigned sampleType)
        : _samplesPerPacket(samplesPerPacket), _subbandsPerPacket(nSubbands),
          _nrPolarisations(nPolarisations),_samplesPerSecond(samplesPerSecond),
          _sampleSize(sampleType)
{   
    // Get configuration options
    _startTime = _startBlockid = 0;
    _packetsAccepted = 0;
    _packetsRejected = 0;

    _packetSize = _subbandsPerPacket * _samplesPerPacket * _nrPolarisations;

    size_t headerSize = sizeof(struct UDPPacket::Header);
    switch (sampleType)
    {
        case 4:
            _packetSize = _packetSize * sizeof(TYPES::i4complex) + headerSize;
            break;
        case 8:
            _packetSize = _packetSize * sizeof(TYPES::i8complex) + headerSize;
            break;
        case 16:
            _packetSize = _packetSize * sizeof(TYPES::i16complex) + headerSize;
            break;
    }
    
    // Initialise chunker
    connectDevice(port);
}

UDPChunker::~UDPChunker() 
{
    // Close socket
    _socket -> close();
}

// Connect socket to start receiving data
void UDPChunker::connectDevice(unsigned port)
{
    _socket = new QUdpSocket;

    if (!_socket -> bind(port)) {
        fprintf(stderr, "UDPChunker::connectDevice(): Unable to bind to UDP port!\n");
        exit(0);
    }
}
    
// Run the UDP receiving thread
void UDPChunker::run()
{
    // Initialise receiving thread
    unsigned prevSeqid = _startTime;
    unsigned prevBlockid = _startBlockid;
    UDPPacket emptyPacket, currPacket;
    generateEmptyPacket(emptyPacket);
    
    unsigned long counter = 0;
    
    // Read in packet forever
    while(true) {
    
         // Wait for datagram to be available.
        while (!_socket -> hasPendingDatagrams())
            _socket -> waitForReadyRead(1);

        if (_socket->readDatagram(reinterpret_cast<char*>(&currPacket), _packetSize) <= 0) {
            printf("Error while receiving UDP Packet!\n");
            continue;
        }
        
        unsigned seqid, blockid;

        // TODO: Check for endianness
        seqid   = currPacket.header.timestamp;
        blockid = currPacket.header.blockSequenceNumber;

        // First time next has been run, initialise startTime and startBlockId
        if (counter == 0 && _startTime == 0) {
            prevSeqid = _startTime = _startTime == 0 ? seqid : _startTime;
            prevBlockid = _startBlockid = _startBlockid == 0 ? blockid : _startBlockid;
            _buffer -> setTimingVariables(seqid + blockid / _samplesPerSecond * 1.0, // timestamp
                                          1 / (_samplesPerSecond * 1.0));              // blockrate
        }

        // Sanity check in seqid. If the seconds counter is 0xFFFFFFFF,
        // the data cannot be trusted (ignore)
        if (seqid == ~0U || prevSeqid + 10 < seqid) {
            ++_packetsRejected;
            continue;
        }

//      Check that the packets are contiguous. Block id increments by no_blocks
//      which is defined in the header. Blockid is reset every interval (although
//      it might not start from 0 as the previous frame might contain data from this one)
        unsigned totBlocks = _samplesPerSecond / _samplesPerPacket;
        unsigned lostPackets = 0, diff = 0;

        diff =  (blockid >= prevBlockid) ? (blockid - prevBlockid) : (blockid + totBlocks - prevBlockid);

        // Duplicated packets... ignore
        if (diff < _samplesPerPacket) { 
            ++_packetsRejected;
            continue;
        }
        
        // Missing packets
        else if (diff > _samplesPerPacket) {
            lostPackets = (diff / _samplesPerPacket) - 1;  // -1 since it includes the received packet as well
            fprintf(stderr, "==================== Generated %u empty packets =====================\n", lostPackets);
        }

        // Generate lostPackets empty packets, if any
        unsigned packetCounter = 0;
        for (packetCounter = 0; packetCounter < lostPackets; ++packetCounter)
        {
            // Generate empty packet with correct seqid and blockid
            prevSeqid = (prevBlockid + _samplesPerPacket < totBlocks) ? prevSeqid : prevSeqid + 1;
            prevBlockid = (prevBlockid + _samplesPerPacket) % totBlocks;
//            emptyPacket.header.timestamp = prevSeqid;
//            emptyPacket.header.blockSequenceNumber = prevBlockid;
            writePacket(emptyPacket);
        }

        counter += packetCounter;

        // Write received packet
        ++_packetsAccepted;
        writePacket(currPacket);
        prevSeqid = seqid;
        prevBlockid = blockid;
    
        counter ++;
        
//        if (counter % 1000000 == 0)
//            printf("====================== Received 1000000 packets ====================\n");
    }
}

// Write packet to CircularBuffer
void inline UDPChunker::writePacket(UDPPacket& packet)
{
    // NOTE:: C stores these value as imaginary most significant!!
    _buffer -> writeData(_samplesPerPacket, _subbandsPerPacket,
                         reinterpret_cast<float *>(packet.data), true);
}
     
// Generate empty packet   
void UDPChunker::generateEmptyPacket(UDPPacket& packet)
{
    size_t size = _packetSize - sizeof(struct UDPPacket::Header);
    memset((void*) packet.data, 0, size);
    packet.header.nrBeamlets = _subbandsPerPacket;
    packet.header.nrBlocks   = _samplesPerPacket;
    packet.header.timestamp  = 0;
    packet.header.blockSequenceNumber = 0;
};
