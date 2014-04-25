#include "DoubleBuffer.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include "time.h"
#include "sys/time.h"

#define SWAP(x,y) unsigned t; t=x; x=y; y=t;
#define HEAP_BUFFERS 8
#define BUSY_WAIT    0.0001

// Class constructor
DoubleBuffer::DoubleBuffer(unsigned nantennas, unsigned nchans, unsigned nsamp) 
    : _nantennas(nantennas), _nchans(nchans), _nsamp(nsamp)
{
    // Initialise buffers
    _buffer    =  (unsigned char **) malloc(2 * sizeof(unsigned char*));
    _buffer[0] =  (unsigned char *) malloc(nantennas * nsamp * nchans * sizeof(unsigned char));
    _buffer[1] =  (unsigned char *) malloc(nantennas * nsamp * nchans * sizeof(unsigned char));

    _readerBuffer = _samplesBuffered = _fullBuffers = 0;
    _writerHeap = _readerHeap = 0;
    _writerBuffer = 1;
    
    printf("============== Buffers Initialised - read: %d, write: %d ==============\n", _readerBuffer, _writerBuffer);

    // Initialise synchronisation objects
    pthread_mutex_init(&_readMutex, NULL);
    pthread_mutex_init(&_writeMutex, NULL);

    // Set thread affinity
    pthread_t thread = pthread_self();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);

    if ((pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset)) != 0)
        perror("Cannot set pthread affinity");
}

// Populate writer parameters (called from network thread)
unsigned char *DoubleBuffer::setHeapParameters(unsigned heapNchans, unsigned heapNsamp)
{
    _heapChans = heapNchans;
    _heapNsamp = heapNsamp;

    // Allocate heap buffers
    // TODO: Contiguous alloc with pointer partitioning?
    _heapBuffers = (unsigned char **) malloc(HEAP_BUFFERS * sizeof(unsigned char *));
    for(unsigned i = 0; i < HEAP_BUFFERS; i++)
        _heapBuffers[i] = (unsigned char *) malloc(_nantennas * heapNchans * heapNsamp * sizeof(unsigned char));

    // Initialise heap variables
    _readerHeap = 0;
    _writerHeap = 0;

    // Return first heap buffer
    return _heapBuffers[_readerHeap];
}

// Lock buffer segment for reading
unsigned char *DoubleBuffer::prepareRead(double *timestamp, double *blockrate)
{
    // Busy wait for enough data, for now
    while (_fullBuffers < 1)
        sleep(BUSY_WAIT);
        
    // Data available
    *timestamp = _timestamp;
    *blockrate = _blockrate;
    return _buffer[_readerBuffer];
}

// Read is ready
void DoubleBuffer::readReady()
{
    // Mutex buffer control, mark buffer as empty
    //_readMutex.lock();
    pthread_mutex_lock(&_readMutex);
    _fullBuffers--;
    pthread_mutex_unlock(&_readMutex);
    //_readMutex.unlock();
    printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
}

// Check whether heap can be written to buffer
unsigned char *DoubleBuffer::writeHeap(double timestamp, double blockrate)
{
    // Reader has filled in it's heap buffer, need to advance to new one
    // unless it's being processed by the writer

    // This is the only place where the _readerHeap is updated, so no need to lock
    _readerHeap = (_readerHeap + 1) % HEAP_BUFFERS;

    // Update timing
    if (_have_timing == false)
    { _timestamp = timestamp; _blockrate = blockrate; _have_timing = true; }

//    _writeMutex.lock();
    pthread_mutex_lock(&_writeMutex);
    while (_readerHeap == _writerHeap)
    {
//        _writeMutex.unlock();
        pthread_mutex_unlock(&_writeMutex);
        sleep(BUSY_WAIT);
        pthread_mutex_lock(&_writeMutex);
//        _writeMutex.lock();
    }
    pthread_mutex_unlock(&_writeMutex);
//    _writeMutex.unlock();

    // Return new heap buffer
    return _heapBuffers[_readerHeap];
}

// Write data to buffer
void DoubleBuffer::run()
{
    // Temporary working buffer for local heap re-organisation
    unsigned char *local_heap = (unsigned char *) malloc(_nantennas * _heapNsamp * sizeof(unsigned char));

    // Infinite loop which read heap data from network thread and write it to buffer
    unsigned counter = 0;
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

        for(unsigned c = 0; c < _heapChans; c++)
        {    
            // Write re-organised channel to double buffer
            memcpy(&_buffer[_writerBuffer][c * _nsamp * _nantennas + _samplesBuffered * _nantennas],
                   local_heap, _nantennas * _heapNsamp * sizeof(unsigned char));

            memcpy(_buffer[_writerBuffer] + c * _nsamp * _nantennas + _samplesBuffered * _nantennas,
                   _heapBuffers[_writerHeap] + c * _nantennas * _heapNsamp, _nantennas * _heapNsamp * sizeof(unsigned char));
       }

        // Increment sample count
        _samplesBuffered += _heapNsamp;

        // Dealing with a new heap, check if buffer is already full
        if (_samplesBuffered == _nsamp)
        {
            // TEMP: Heap is ready... dump to disk
//            if (counter > 1)
//            {
//                FILE *fp = fopen("heap_dump.dat", "wb");
//                fwrite(_buffer[_writerBuffer], sizeof(unsigned char), _nsamp * _nantennas * _heapChans, fp);
//                fclose(fp);
//                printf("Written buffer to disk\n");
//                sleep(60);Mutex
//                exit(0);
//            }

            // Check if reading buffer has been read
            while(_fullBuffers == 1)
                sleep(BUSY_WAIT);
        
            // Lock critical section with mutex, and swap buffers
//            _readMutex.lock();
            pthread_mutex_lock(&_readMutex);
            _samplesBuffered = 0; _fullBuffers++;
            SWAP(_writerBuffer, _readerBuffer)
            pthread_mutex_unlock(&_readMutex);
//            _readMutex.unlock();
            printf("========================== Full Buffers: %d ==========================\n", _fullBuffers);
            counter++;
        }

        // Finished writing heap
//        _writeMutex.lock();
        pthread_mutex_lock(&_writeMutex);
        if (_samplesBuffered == 0)
            _have_timing = 0;
        _writerHeap = (_writerHeap + 1) % HEAP_BUFFERS;
        pthread_mutex_unlock(&_writeMutex);
//        _writeMutex.unlock();
    }
}
