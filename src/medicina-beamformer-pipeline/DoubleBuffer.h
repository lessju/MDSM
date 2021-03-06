// A 2D circular buffer for buffering input data (time vs frequency)
// Data stored as complex 16-floats OR 32-bit floats

#ifndef DoubleBuffer_H
#define DoubleBuffer_H

#include <QMutex>
#include <QThread>
#include <pthread.h>

class DoubleBuffer: public QThread 
{

    public:
        DoubleBuffer(unsigned nbeams, unsigned nchans, unsigned nsamp);
        ~DoubleBuffer() { }    
        
        // Notify that a read request has finished
        void readReady();

        // Check whether heap can be written to buffer
        unsigned char *writeHeap(double timestamp, double blockrate);

        // Wait for a buffer to become available
        unsigned char *prepareRead(double *timestamp, double *blockrate);

        // Populate writer parameters (called from network thread)
        unsigned char *setHeapParameters(unsigned nchans, unsigned nsamp);
    
        // Set timing variable (called from network thread)
        void  setTimingVariables(double timestamp, double blockrate);

        // Return heap buffer allocated here
        unsigned char  *getHeapBuffer();

        // Infinite thread loop to populate buffers
        virtual void run();    

    private:
        // Double buffer
        unsigned char     **_buffer;

        // Heap buffer
        unsigned char      **_heapBuffers;
        
        // Input data parameters
        unsigned  _nantennas, _nchans, _nsamp, ;

        // Buffer parameters
        unsigned  _fullBuffers, _samplesBuffered;
        unsigned _readerBuffer, _writerBuffer;

        // Heap parameters and variables
        unsigned _heapChans, _heapNsamp;
        unsigned _readerHeap, _writerHeap;

        // Variables dictating whether we are copying voltages and whether we have timing
        double    _timestamp,_blockrate;
        bool      _have_timing;

        pthread_mutex_t _readMutex, _writeMutex; 
        // QMutex    _readMutex, _writeMutex; 
};

#endif // DoubleBuffer_H
