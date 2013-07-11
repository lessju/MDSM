#include "DoubleBuffer.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include "time.h"
#include "sys/time.h"

#define SWAP(x,y) unsigned t; t=x; x=y; y=t;
#define HEAP_BUFFERS 4
#define BUSY_WAIT    0.1

// Class constructor
DoubleBuffer::DoubleBuffer(unsigned nantennas, unsigned nchans, unsigned nsamp) 
    : _nantennas(nantennas), _nchans(nchans), _nsamp(nsamp)
{
    // Initialise buffers
    // TODO: Allocate these with CUDA
    _buffer    =  (char **) malloc(2 * sizeof(char*));
    _buffer[0] =  (char *) malloc(nantennas * nsamp * nchans * sizeof(char));
    _buffer[1] =  (char *) malloc(nantennas * nsamp * nchans * sizeof(char));

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
        _heapBuffers[i] = (char *) malloc(_nantennas * heapNchans * heapNsamp * sizeof(char));

    // Initialise heap variables
    _readerHeap = 0;
    _writerHeap = 0;

    // Return first heap buffer
    return _heapBuffers[_readerHeap];
}

// Lock buffer segment for reading
char *DoubleBuffer::prepareRead(double *timestamp, double *blockrate)
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
    // Temporary working buffer for local heap re-organisation
    unsigned char *local_heap = (unsigned char *) malloc(_nantennas * _heapNsamp * sizeof(unsigned char));

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

        // Heap is in channel/spectra/antenna order. For each channel:
        // A0x[0]A0y[1]   ... A0x[127]A0y[127]
        //                ...
        // A15x[0]A15y[1] ... A15x[127]A15y[127]
        // 8-bits per value (4-bit real, 4-bit imaginary)

//        for(unsigned i = 0; i < 4096 * 1024; i++)
//            printf("%d ", (unsigned char) _heapBuffers[_writerHeap][i]);
 
        for(unsigned c = 0; c < _heapChans; c++)
        {
            // Loop over groups of antennas to populate local heap buffer
            for(unsigned a = 0; a < _nantennas / 2; a++)
                for(unsigned s = 0; s < _heapNsamp; s++)
                {
                    local_heap[s * _nantennas + a * 2]     = (unsigned char) _heapBuffers[_writerHeap][c * _nantennas * _heapNsamp + a * _heapNsamp * 2 + s * 2];
                    local_heap[s * _nantennas + a * 2 + 1] = (unsigned char) _heapBuffers[_writerHeap][c * _nantennas * _heapNsamp + a * _heapNsamp * 2 + s * 2 + 1];
                }

            // Write re-organised channel to double buffer
//            memcpy(&_buffer[_writerBuffer][c * _nsamp * _nantennas + _samplesBuffered * _nantennas],
//                   local_heap, _nantennas * _heapNsamp * sizeof(unsigned char));
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
