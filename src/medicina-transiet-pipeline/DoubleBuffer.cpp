#include "DoubleBuffer.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include "time.h"
#include "sys/time.h"

#define SWAP(x,y) unsigned t; t=x; x=y; y=t;
#define PWR(X,Y) X*X + Y*Y
#define HEAP_BUFFERS 4
#define BUSY_WAIT    0.001

// Class constructor
DoubleBuffer::DoubleBuffer(unsigned nbeams, unsigned nchans, unsigned nsamp, bool voltage) 
    : _nbeams(nbeams), _nchans(nchans), _nsamp(nsamp), _copy_voltage(voltage)
{
    // Initialise buffers
    // TODO: Allocate these with CUDA
    _buffer    =  (float **) malloc(2 * sizeof(float*));
    _buffer[0] =  (float *) malloc(nbeams * nsamp * nchans * sizeof(float));
    _buffer[1] =  (float *) malloc(nbeams * nsamp * nchans * sizeof(float));

    _readerBuffer = _samplesBuffered = _fullBuffers = 0;
    _writerHeap = _readerHeap = 0;
    _writerBuffer = 1;
    
    printf("============== Buffers Initialised - read: %d, write: %d ==============\n", _readerBuffer, _writerBuffer);
}

// Populate writer parameters (called from network thread)
char *DoubleBuffer::setHeapParameters(unsigned heapNchans, unsigned heapNsamp)
{
    _heapChans = heapNchans;
    _heapNsamp = heapNsamp;

    // Allocate heap buffers
    // TODO: Contiguous alloc with pointer partitioning?
    _heapBuffers = (char **) malloc(HEAP_BUFFERS * sizeof(char *));
    for(unsigned i = 0; i < HEAP_BUFFERS; i++)
        _heapBuffers[i] = (char *) malloc(_nbeams * heapNchans * heapNsamp * sizeof(float));

    // Initialise heap variables
    _readerHeap = 0;
    _writerHeap = 0;

    // Return first heap buffer
    return _heapBuffers[_readerHeap];
}

// Lock buffer segment for reading
float *DoubleBuffer::prepareRead(double *timestamp, double *blockrate)
{
    // Busy wait for enough data, for now
    while (_fullBuffers < 1)
        sleep(BUSY_WAIT);
        
    // Data available (TODO: why are we mutex locking here?)
    *timestamp = _timestamp;
    *blockrate = _blockrate;
    return _buffer[_readerBuffer];
}

// Read is ready
void DoubleBuffer::readReady()
{
    // Mutex buffer control, mark buffer as empty
    _readMutex.lock();
    _fullBuffers--;
    _readMutex.unlock();
    printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
}

// Check whether heap can be written to buffer
char *DoubleBuffer::writeHeap(double timestamp, double blockrate)
{
    // Reader has filled in it's heap buffer, need to advance to new one
    // unless it's being processed by the writer

    // This is the only place where the _readerHeap is updates, so no need to lock
    _readerHeap = (_readerHeap + 1) % HEAP_BUFFERS;

    // Update timing
    if (_have_timing == false)
    { _timestamp = timestamp; _blockrate = blockrate; _have_timing = true; }

    _writeMutex.lock();
    while (_readerHeap == _writerHeap)
    {
        _writeMutex.unlock();
        sleep(BUSY_WAIT);
        _writeMutex.lock();
    }
    _writeMutex.unlock();

    // Return new heap buffer
    return _heapBuffers[_readerHeap];
}

// Write data to buffer
void DoubleBuffer::run()
{
    // Infinite loop which read heap data from network thread and write it to buffer
    while(true)
    {
        // Wait for buffer to be read
        while(_fullBuffers == 2)
            sleep(BUSY_WAIT);

        // Wait for heap data to become available
        while (_writerHeap == _readerHeap)
            sleep(BUSY_WAIT);

        // Reader has advanced one buffer, we can start writing current buffer

        // Store incoming data into current writable double buffer
        if (_copy_voltage)
            for(unsigned b = 0; b < _nbeams; b++)
                for(unsigned c = 0; c < _heapChans; c++)
                    memcpy(&_buffer[_writerBuffer][b * _nchans * _nsamp + c * _nsamp + _samplesBuffered],
                           &_heapBuffers[_writerHeap][b * _heapChans * _heapNsamp + c * _heapNsamp],
                           _heapNsamp * sizeof(float));

        else
        {
            short *complexData = (short *) _heapBuffers[_writerHeap];
            for(unsigned b = 0; b < _nbeams; b++)
                for(unsigned c = 0; c < _heapChans; c++)
                    for(unsigned s = 0; s < _heapNsamp; s++)
                    {
                        unsigned bufferIndex = b * _nchans * _nsamp + c * _nsamp + s + _samplesBuffered;
                        unsigned complexIndex = 2 * (b * _heapChans * _heapNsamp + c * _heapNsamp + s);
                        _buffer[_writerBuffer][bufferIndex] =  PWR(complexData[complexIndex], 
                                                                complexData[complexIndex+1]);
                    }
        }

        // Increment sample count
        _samplesBuffered += _heapNsamp;

        // Dealing with a new heap, check if buffer is already full
        if (_samplesBuffered == _nsamp)
        {
            // Check if reading buffer has been read
            while(_fullBuffers == 1)
                sleep(BUSY_WAIT);
        
            // Lock critical section with mutex, and swap buffers
            _readMutex.lock();
            _samplesBuffered = 0; _fullBuffers++;
            SWAP(_writerBuffer, _readerBuffer)
            printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
            _readMutex.unlock();
        }

        // Finished writing heap
        _writeMutex.lock();
        if (_samplesBuffered == 0)
            _have_timing = 0;
        _writerHeap = (_writerHeap + 1) % HEAP_BUFFERS;
        _writeMutex.unlock();
    }
}
