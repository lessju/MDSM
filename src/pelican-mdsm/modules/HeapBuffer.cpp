#include "HeapBuffer.h"

HeapBuffer::HeapBuffer(unsigned numBeams, unsigned numHeaps, unsigned numFreqs,
                       unsigned numSpectra, unsigned packetsPerHeap, unsigned port)
  : _nBeams(numBeams), _nHeaps(numHeaps), _nFreqs(numFreqs),
    _nSpectra(numSpectra), _nPackets(packetsPerHeap), _port(port)
{
    // Setup connection
    _socket = 1;
    _bufferSize = 0;
    setupReceiver();

    // Allocate buffers and heaps
    allocateBuffers();

    // Allocate reader/writer pointers
    _writerId = _writerHeap = _readerId = 0;
}

// Setup receiver for continuous 
void HeapBuffer::setupReceiver()
{
    _socket = socket(PF_INET, SOCK_DGRAM, 0); // create a new UDP socket descriptor

    if (_socket == -1)
    {
        std::cerr << "Unable to create UDP socket" << std::endl;              
        exit(-1);
    }

    // initialize local address struct
    _serverAddress.sin_family = AF_INET; // host byte order
    _serverAddress.sin_port = htons(_port); // short, network byte order
    _serverAddress.sin_addr.s_addr = htonl(INADDR_ANY); // listen on all interfaces
    memset(_serverAddress.sin_zero, 0, sizeof(_serverAddress.sin_zero));

    // Bind socket to local address
    if (bind(_socket, (sockaddr *) &_serverAddress, sizeof(_serverAddress)) == -1)
    {
        std::cerr << "Unable to bind socket" << std::endl;
        exit(-1);
    }

    // Prevent "address already in use" errors
    const int on = 1;
    if (setsockopt(_socket, SOL_SOCKET, SO_REUSEADDR, (void *)&on, sizeof(on)) == -1)
    {
        std::cout << "Could not re-use address" << std::endl;
        exit(-1);
    }

    // Set socket options (buffer size)
    if (_bufferSize != 0)
    {
        if (setsockopt(_socket, SOL_SOCKET, SO_RCVBUF, &_bufferSize, sizeof(_bufferSize)) != 0)
        {
            std::cerr << "Could not set buffer receive size: " << strerror(errno) << std::endl;
            exit(-1);
        }
    }
}

// Start receiver loop
void HeapBuffer::run()
{
    // Temporary packet buffer
    char *packet = (char *) malloc(SPEAD_MAX_PACKET_LEN);

    socklen_t addr_len = sizeof(struct sockaddr);
    unsigned counter;

    while (true)
    {
        // Receive UDP packet
        if (recvfrom(_socket, packet, SPEAD_MAX_PACKET_LEN, 0,
                    (struct sockaddr *) &_serverAddress, &addr_len) <= 0) 
        {
            printf("Error while receiving UDP Packet\n");
            continue;
        }

        unsigned nItems;
        char *payload;

        // Unpack packet header (64 bits)
        uint64_t hdr;
        hdr = SPEAD_HEADER(packet);

        if ((SPEAD_GET_MAGIC(hdr) != SPEAD_MAGIC) ||
                (SPEAD_GET_VERSION(hdr) != SPEAD_VERSION) ||
                (SPEAD_GET_ITEMSIZE(hdr) != SPEAD_ITEM_PTR_WIDTH) ||
                (SPEAD_GET_ADDRSIZE(hdr) != SPEAD_HEAP_ADDR_WIDTH))
            continue;

        nItems = SPEAD_GET_NITEMS(hdr);
        payload = packet + SPEAD_HEADERLEN + nItems * SPEAD_ITEMLEN;

        // Unpack packet items: Each item is 64 bits wide and all beam items use direct mode addressing
        uint64_t heapNumber, heapSize, payloadOffset, payloadLen, beamId, item;

        // Item 1: heap number
        item = SPEAD_ITEM(packet, 1);
        heapNumber = SPEAD_ITEM_ADDR(item);

        // Item 2: heap size
        item = SPEAD_ITEM(packet, 2);
        heapSize = SPEAD_ITEM_ADDR(item);

        // Item 3: payload length
        item = SPEAD_ITEM(packet, 3);
        payloadOffset = SPEAD_ITEM_ADDR(item);

        // item 4: item number
        item = SPEAD_ITEM(packet, 4);
        payloadLen = SPEAD_ITEM_ADDR(item);

        // Item 5: data
        item = SPEAD_ITEM(packet, 5);
        beamId = SPEAD_ITEM_ID(item) - 6000 * 8; // -6000 * 8 for beam id .... for some reason
        
        // Add packet to heap buffer
        addPacket(beamId, heapNumber, payloadOffset, payloadLen, packet + sizeof(uint64_t) * 6);

        counter++;
    }
}

// Request next ready heap. Blocking call
Heap *HeapBuffer::getHeap()
{
    // Get pointer to required buffer
    Heap *heap = &_buffer[_readerId];

    while (true)
    {
        pthread_mutex_lock(&(heap -> mutex));
        unsigned ready = heap -> ready;
        pthread_mutex_unlock(&(heap -> mutex));

        if (ready != 1 )
        {
            usleep(100); // Busy wait   
            continue;
        }
        else
            break;
    } 

    // Data available
    return heap;
}

// Set heap as ready
void HeapBuffer::heapReady()
{
    // Update heap status
    pthread_mutex_lock(&(_buffer[_readerId].mutex));

    // Heap processed, reset
    resetHeap(&_buffer[_readerId], 0, 2);

    pthread_mutex_unlock(&(_buffer[_readerId].mutex));

    // Update reader id for current beam
    _readerId = (_readerId + 1 >= _nHeaps) ? 0 : _readerId + 1;   
}

// Add packet to heap
void HeapBuffer::addPacket(unsigned beamId, unsigned heapNumber, unsigned heapOffset, unsigned size, char *data)
{
    Heap *heap = &_buffer[_writerId];

    // Check if this belongs to current heap
    if (heapNumber == _writerHeap)
    {
        // Copy packet data to buffer
        memcpy(((char *) heap -> beams[beamId]) + heapOffset, data, size);

        heap -> packets++;

        // Check if heap is ready, if so update status
        if (heap -> packets == _nPackets)
        {
            pthread_mutex_lock(&(heap -> mutex));
            std::cout << "Heap fully loaded" << std::endl;
            heap -> ready = 1;
            pthread_mutex_unlock(&(heap ->mutex));
        }

        return;
    }

    // Not in current heap, check if it belongs to a previous heap (if so ignore)
    if (heapNumber < _writerHeap) return;

    // Check if previous heap has was properly processd (due to lost packets)
    // NOTE: update for scenario where we lose many packets
    if (heap -> packets < _nPackets && heap -> packets != 0)
    {
        // Previous heap was not filled up, set as ready
        pthread_mutex_lock(&(heap -> mutex));
        heap -> ready = 1;
        pthread_mutex_unlock(&(heap -> mutex));
        std::cout << "Heap partially ready: " << heapNumber << ", " << heap -> packets << std::endl;
    }

    // Packet belongs to a new heap, update pointer and get new heap
    _writerHeap = heapNumber;
    _writerId = (_writerId + 1 >= _nHeaps) ? 0 : _writerId + 1;
    heap = &_buffer[_writerId];

    // Check if new heap is already processed, if not then busy wait
    while (true)
    {
        pthread_mutex_lock(&(heap -> mutex));
        if (heap -> ready != 2)
        {
            pthread_mutex_unlock(&(heap -> mutex));
            usleep(10);
            continue;
        }   
        else
        {
            heap -> ready = 0;
            pthread_mutex_unlock(&(heap -> mutex));
            break;
        }
    }

    // Copy packet contents to new heap
    memcpy(((char *) heap -> beams[beamId]) + heapOffset, data, size);
    heap -> packets += 1;
}

// Allocate heap buffers
void HeapBuffer::allocateBuffers()
{
    // Allocate buffer memory
    _buffer = (Heap *) malloc(_nHeaps * sizeof(Heap));

    // Allocate beam memory in each heap
    for(unsigned i = 0; i < _nHeaps; i++)
    {
        // Initialise heap mutex object
        pthread_mutex_init(&(_buffer[i].mutex), NULL);

        // Allocate beams buffers as one contiguous block and lock in memory
        _buffer[i].beams = (Beam *) malloc(_nBeams * sizeof(Beam));
        char *ptr = (char *) malloc(_nBeams * _nFreqs * _nSpectra * sizeof(i16complex));
        mlock(ptr, _nBeams * _nFreqs * _nSpectra * sizeof(i16complex));

        // Allocate data memory for each beam
        for(unsigned j = 0; j < _nBeams; j++)
              _buffer[i].beams[j] = (i16complex *) (ptr + j * _nFreqs * _nSpectra * sizeof(i16complex));
    
        // Initialise heap
        resetHeap(&_buffer[i], 0, 2);
    }

    std::cout << "Initialised heap buffer" << std::endl;
}

// Reset heap
void HeapBuffer::resetHeap(Heap *heap, unsigned id, unsigned ready)
{
    heap -> id = id;
    heap -> ready = ready;
    heap -> packets = 0;

    for(unsigned i = 0; i < _nBeams; i++)
        memset(heap -> beams[i], 0, _nSpectra * _nFreqs * sizeof(i16complex));
}
