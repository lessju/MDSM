#ifndef PacketChunker_H
#define PacketChunker_H

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
                         unsigned samplesPerSecond, unsigned packetsPerHeap);
        ~PacketChunker();
        
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
        unsigned _numberOfAntennas;
        unsigned _samplesPerSecond;
        unsigned _startTime;
        unsigned _startBlockid;
        unsigned _packetsPerHeap;
        unsigned _heapSize;
};

#endif // PACKET_CHUNKER_H
