#ifndef UDPCHUNKER_H
#define UDPCHUNKER_H

#include "DoubleBuffer.h"
#include "UdpHeader.h"

#include <QtCore/QString>
#include <QtCore/QObject>
#include <QUdpSocket>
#include <QThread>

class UDPChunker: public QThread
{

    public:
        UDPChunker(unsigned port, unsigned samplesPerPacket, unsigned nSubbands, 
                       unsigned nPolarisations, unsigned samplesPerSecond, unsigned sampleType);
        ~UDPChunker();
        
        void setDoubleBuffer(DoubleBuffer *buffer) { _buffer = buffer; }

        virtual void run();
        
    private:
        /// Generates an empty UDP packet.
        void generateEmptyPacket(UDPPacket& packet);
        void connectDevice(unsigned port);
        void inline writePacket(UDPPacket& packet);

    private:
        DoubleBuffer   *_buffer;
        QUdpSocket       *_socket;

        unsigned _nPackets;
        unsigned _packetsRejected;
        unsigned _packetsAccepted;
        unsigned _samplesPerPacket;
        unsigned _subbandsPerPacket;
        unsigned _nrPolarisations;
        unsigned _samplesPerSecond;
        unsigned _startTime;
        unsigned _startBlockid;
        unsigned _packetSize;
        unsigned _sampleSize;

};

#endif // UDPCHUNKER_H
