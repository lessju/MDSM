// A 2D circular buffer for buffering input data (time vs frequency)
// Data stored as complex 16-floats OR 32-bit floats

#ifndef DoubleBuffer_H
#define DoubleBuffer_H

#include <QMutex>

class DoubleBuffer {

    public:
        DoubleBuffer(unsigned nsamp, unsigned nchans, unsigned npols);
        ~DoubleBuffer() { }    
        
        void readReady();
        float * prepareRead(double *timestamp, double *blockrate);
        void writeData(unsigned nsamp, unsigned nchans, float* data, bool interleavedMode);
        void setTimingVariables(double timestamp, double blockrate);
    
    private:
        // Buffer
        float        **_buffer;
    
        unsigned     _nsamp, _nchans, _npols, _readBuff, _writeBuff, _counter;
        unsigned     _writePtr, _buffLen, _fullBuffers, _samplesBuffered;
        double       _timestamp,_blockrate;
        
        QMutex       _mutex; 
};

#endif // DoubleBuffer_H
