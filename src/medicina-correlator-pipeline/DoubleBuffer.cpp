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

        // Data format is channel/antenna/spectra, where 2 antennas are interleaved
        // [[chan1 [A|B|A|B...|B][C|D|C|D...|D] ][chan2 ...] ... [chanN]]
        // Output format data format will be: channel - antenna - spectra (so will probably need)
        // a transpose for each channel for optimised correlator
        for(unsigned c = 0; c < _heapChans; c++)
            for(unsigned a = 0; a < _nantennas; a += 2)
            {
                unsigned ant1Index = c * _nantennas * _nsamp + a     * _nsamp + _samplesBuffered;
                unsigned ant2Index = c * _nantennas * _nsamp + (a+1) * _nsamp + _samplesBuffered;
                unsigned heapIndex = c * _nantennas * _heapNsamp + a * _heapNsamp;
                for(unsigned s = 0; s < _heapNsamp; s++)
                {
                    _buffer[_writerBuffer][ant1Index + s] = _heapBuffers[_writerHeap][heapIndex + s * 2];
                    _buffer[_writerBuffer][ant2Index + s] = _heapBuffers[_writerHeap][heapIndex + s * 2 + 1];
                }
            }

        // Increment sample count
        _samplesBuffered += _heapNsamp;
        printf("%d\n", _samplesBuffered);

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
