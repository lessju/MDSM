#ifndef SPEADBEAMCHUNKER_H
#define SPEADBEAMCHUNKER_H

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <complex>

// SPEAH-specific defines

#if defined(__x86_64__) && !defined(__NR_recvmmsg)
#define __NR_recvmmsg    299
#endif

#ifndef htonll
#ifdef _BIG_ENDIAN
#define htonll(x)   ((uint64_t)x)
#define ntohll(x)   ((uint64_t)x)
#else
#define htonll(x)   ((((uint64_t)htonl(x)) << 32) + htonl(((uint64_t)x) >> 32))
#define ntohll(x)   ((((uint64_t)ntohl(x)) << 32) + ntohl(((uint64_t)x) >> 32))
#endif
#endif

// Flavor constants
#define SPEAD_MAGIC                 0x53
#define SPEAD_VERSION               4
#define SPEAD_ITEMSIZE              64
#define SPEAD_ADDRSIZE              40
#define SPEAD_HEAP_ADDR_WIDTH     (SPEAD_ADDRSIZE/8)
#define SPEAD_ITEM_PTR_WIDTH	  ((SPEAD_ITEMSIZE-SPEAD_ADDRSIZE)/8)
#define SPEAD_ITEMLEN             (SPEAD_ITEMSIZE/8)
#define SPEAD_ADDRLEN             (SPEAD_ADDRSIZE/8)

#define SPEAD_ITEMMASK              0xFFFFFFFFFFFFFFFFLL
#define SPEAD_ADDRMASK              (SPEAD_ITEMMASK >> (SPEAD_ITEMSIZE-SPEAD_ADDRSIZE))
#define SPEAD_IDMASK                (SPEAD_ITEMMASK >> (SPEAD_ADDRSIZE+1))
#define SPEAD_ADDRMODEMASK          0x1LL
#define SPEAD_DIRECTADDR            0
#define SPEAD_IMMEDIATEADDR         1

#define SPEAD_MAX_PACKET_LEN       9200
#define SPEAD_MAX_FMT_LEN          1024

// Reserved Item IDs
#define SPEAD_HEAP_CNT_ID           0x01
#define SPEAD_HEAP_LEN_ID           0x02
#define SPEAD_PAYLOAD_OFF_ID     0x03
#define SPEAD_PAYLOAD_LEN_ID     0x04
#define SPEAD_DESCRIPTOR_ID         0x05
#define SPEAD_STREAM_CTRL_ID        0x06

#define SPEAD_STREAM_CTRL_TERM_VAL  0x02
#define SPEAD_ERR                   -1
#define SPEAD_SUCCESS               1

// Header Macros
#define SPEAD_HEADERLEN             8
#define SPEAD_HEADER(data) (ntohll(((uint64_t *)(data))[0]))
#define SPEAD_HEADER_BUILD(nitems) ((((uint64_t) SPEAD_MAGIC) << 56) | (((uint64_t) SPEAD_VERSION) << 48) | (((uint64_t) SPEAD_ITEM_PTR_WIDTH) << 40) | (((uint64_t) SPEAD_HEAP_ADDR_WIDTH) << 32) | (0xFFFFLL & (nitems)))
#define SPEAD_GET_MAGIC(hdr) (0xFF & ((hdr) >> 56))
#define SPEAD_GET_VERSION(hdr) (0xFF & ((hdr) >> 48))
#define SPEAD_GET_ITEMSIZE(hdr) (0xFF & ((hdr) >> 40))
#define SPEAD_GET_ADDRSIZE(hdr) (0xFF & ((hdr) >> 32))
#define SPEAD_GET_NITEMS(hdr) ((int) 0xFFFF & (hdr))

// ItemPointer Macros
#define SPEAD_ITEM_BUILD(mode,id,val) (((SPEAD_ADDRMODEMASK & (mode)) << (SPEAD_ITEMSIZE-1)) | ((SPEAD_IDMASK & (id)) << (SPEAD_ADDRSIZE)) | (SPEAD_ADDRMASK & (val)))
#define SPEAD_ITEM(data,n) (ntohll(((uint64_t *)(data + (n) * SPEAD_ITEMLEN))[0]))
#define SPEAD_ITEM_MODE(item) ((int)(SPEAD_ADDRMODEMASK & (item >> (SPEAD_ITEMSIZE-1))))
#define SPEAD_ITEM_ID(item) ((int)(SPEAD_IDMASK & (item >> SPEAD_ADDRSIZE)))
#define SPEAD_ITEM_ADDR(item) ((uint64_t)(SPEAD_ADDRMASK & item))
#define SPEAD_SET_ITEM(data,n,pkitem) (((uint64_t *)(data + (n) * SPEAD_ITEMLEN))[0] = htonll(pkitem))

// Format Macros
#define SPEAD_FMT_LEN             4
#define SPEAD_FMT(data,n) (ntohl(((uint32_t *)(data + (n) * SPEAD_FMT_LEN))[0]))
#define SPEAD_FMT_GET_TYPE(fmt) ((char) 0xFF & (fmt >> 24))
#define SPEAD_FMT_GET_NBITS(fmt) ((int) 0xFFFFFF & fmt)

#define SPEAD_U8_ALIGN(data,off) \
    ((off == 0) ? \
    ((uint8_t *)data)[0] : \
    (((uint8_t *)data)[0] << off) | (((uint8_t *)data)[1] >> (8*sizeof(uint8_t) - off)))


#include "DoubleBuffer.h"
#include <QtCore/QString>
#include <QtCore/QObject>
#include <QUdpSocket>
#include <QThread>
#include <iostream>

using namespace std;

class SpeadBeamChunker: public QThread
{
    public:
        SpeadBeamChunker(unsigned port, unsigned nBeams, unsigned nSubbands, unsigned nSpectra, 
                         unsigned samplesPerSecond, unsigned packetsPerHeap);
        ~SpeadBeamChunker();
        
        // Set double buffer pointer
        void setDoubleBuffer(DoubleBuffer *buffer);

        // Start reading UDP packets
        virtual void run();
        
    private:
        /// Generates an empty UDP packet.
        void connectDevice(unsigned port);

    private:
        DoubleBuffer   *_buffer;
        QUdpSocket     *_socket;

        // Heap buffer
        char     *_heap;
        unsigned _samplesPerSubband;
        unsigned _subbandsPerHeap;
        unsigned _numberOfBeams;
        unsigned _samplesPerSecond;
        unsigned _startTime;
        unsigned _startBlockid;
        unsigned _packetsPerHeap;
        unsigned _heapSize;
};

#endif // SPEAD_BEAM_CHUNKER_H
