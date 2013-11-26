#ifndef PacketChunker_H
#define PacketChunker_H

#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/poll.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>

#include "DoubleBuffer.h"
#include <QtCore/QString>
#include <QtCore/QObject>
#include <QUdpSocket>
#include <QThread>
#include <iostream>

using namespace std;

class PacketChunker: public QThread
{
    public:
        PacketChunker(unsigned port, unsigned nAntennas, unsigned nSubbands, unsigned nSpectra, 
                      unsigned packetsPerHeap);
        ~PacketChunker();
        
        // Set double buffer pointer
        void setDoubleBuffer(DoubleBuffer *buffer);

        // Start reading UDP packets
        virtual void run();
        
    private:
        /// Generates an empty UDP packet.
        void connectDevice();

    private:
        DoubleBuffer   *_buffer;

        // Connection and PACKET_MMAP-related variables
        struct iovec *_ring;       // Pointer to iovec aray for easy frame gather
        char         *_map;        // Pointer to memory-mapped ring buffer
        int          _socket;
        short        _port; 
        unsigned     _nframes;

        // Heap buffer
        unsigned char     *_heap;
        unsigned _nsamp;
        unsigned _nchans;
        unsigned _npackets;
        unsigned _nantennas;
        unsigned _startTime;
        unsigned _startBlockid;;
        unsigned _heapSize;
};

#endif // PACKET_CHUNKER_H
