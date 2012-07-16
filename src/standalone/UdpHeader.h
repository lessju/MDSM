#ifndef UDPHEADER_H_
#define UDPHEADER_H_

/**
 * @struct UDPPacket.
 *
 * @details
 * UDP packet data structure.
 *
 * @note
 * All data is in Little Endian format!
 */

struct UDPPacket {
    struct Header {
            unsigned char  version;
            unsigned char  nrBlocks;
            unsigned short configuration;
            unsigned short station;
            unsigned short  nrBeamlets;
            unsigned int timestamp;
            unsigned int blockSequenceNumber;
    } header;

    char data[8130];
};

#endif // UDPHEADER_H_
