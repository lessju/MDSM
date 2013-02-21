// A 2D circular buffer for buffering input data (time vs frequency)
// Data stored as complex 16-floats OR 32-bit floats

#ifndef DoubleBuffer_H
#define DoubleBuffer_H

#include <QMutex>
#include <QThread>

class DoubleBuffer: public QThread 
{

    public:
        DoubleBuffer(unsigned nbeams, unsigned nchans, unsigned nsamp, bool voltage);
        ~DoubleBuffer() { }    
        
        // Notify that a read request has finished
        void readReady();

        // Check whether heap can be written to buffer
        char *writeHeap(double timestamp, double blockrate);

        // Wait for a buffer to become available
        float *prepareRead(double *timestamp, double *blockrate);

        // Populate writer parameters (called from network thread)
        char *setHeapParameters(unsigned nchans, unsigned nsamp);
    
        // Set timing variable (called from network thread)
        void  setTimingVariables(double timestamp, double blockrate);

        // Return heap buffer allocated here
        char  *getHeapBuffer();

        // Infinite thread loop to populate buffers
        virtual void run();    

    private:
        // Double buffer
        float     **_buffer;

        // Heap buffer
        char      **_heapBuffers;
        
        // Input data parameters
        unsigned  _nbeams, _nchans, _nsamp, ;

        // Buffer parameters
        unsigned  _fullBuffers, _samplesBuffered;
        unsigned _readerBuffer, _writerBuffer;

        // Heap parameters and variables
        unsigned _heapChans, _heapNsamp;
        unsigned _readerHeap, _writerHeap;

        // Variables dictating whether we are copying voltages and whether we have timing
        double    _timestamp,_blockrate;
        bool      _copy_voltage, _have_timing;
        
        QMutex    _readMutex, _writeMutex; 
};

#endif // DoubleBuffer_H
