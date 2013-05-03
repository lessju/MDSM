#include "PacketChunker.h"
#include "stdio.h"
#include "stdlib.h"
#include "pthread.h"
#include "sys/time.h"

// We need further arguments:
// - port
// - interface name
// - source IP (for IP-level checks)

// NOTE: These are hardcoded values which depend on the backend F-engine design
// which is sending data throug 10GigE-interface using a custom packet-format
#define PACKET_DATA_LEN    256 // bytes
#define PACKET_HEADER_LEN  8   // bytes

// Maximum size allocatable by kmalloc
#define SIZE_MAX     131072

// Packet offset within frame
#define FRAME_OFFSET      (TPACKET_ALIGN(sizeof(struct tpacket_hdr)) + \
                           TPACKET_ALIGN(sizeof(struct sockaddr_ll))) + 2  // Why +2, alignment?

PacketChunker::PacketChunker(unsigned port, unsigned nAntennas, unsigned nSubbands, 
                             unsigned nSpectra, unsigned packetsPerHeap)
        : _port(port), _nsamp(nSpectra), _nchans(nSubbands), _npackets(packetsPerHeap), _nantennas(nAntennas)
{   
    // Set configuration options
    _startTime = _startBlockid = 0;
    _heapSize  = nAntennas * nSubbands * nSpectra * sizeof(char);

    // Initialise chunker
    connectDevice();
}

PacketChunker::~PacketChunker() 
{
    // Close socket
    close(_socket);
}

// Connect socket to start receiving data
void PacketChunker::connectDevice()
{
    // Create socket
    if ((_socket = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP))) < 0)
    {
        perror("socket() [Need root priviliges]");
        exit(-1);
    }

    // Copy the interface name to ifreq structure
    struct ifreq s_ifr;
    strncpy(s_ifr.ifr_name, "eth0", sizeof(s_ifr.ifr_name));

    // Get interface index
    if(ioctl(_socket, SIOCGIFINDEX, &s_ifr) < 0)
    {
        perror("Couldn't get interface ID");
        exit(-1);
    }

    // Fill sockaddr_ll struct to prepare for binding
    struct sockaddr_ll address;
    address.sll_family   = AF_PACKET;
    address.sll_protocol = 0x08;
    address.sll_ifindex  = s_ifr.ifr_ifindex;
    address.sll_hatype   = 0;
    address.sll_pkttype  = 0;
    address.sll_halen    = 0; 

    // bind socket to eth0
    if(bind(_socket, (struct sockaddr *) &address, sizeof(struct sockaddr_ll)) < 0)
    {
        perror("bind()");
        exit(-1);
    }

    // Get page size (in bytes) to calculate tpacket_req parameters
    long page_size = sysconf(_SC_PAGESIZE);

    // Set up PACKET_MMAP capturing mode (hard-coded values for now)
    struct tpacket_req req;
    req.tp_block_size = page_size;
    req.tp_block_nr   = SIZE_MAX / sizeof(void *);
    req.tp_frame_size = 512;  // Hard-coded for our needs, and is a divisor of pagesize()
    req.tp_frame_nr   = req.tp_block_nr * req.tp_block_size / req.tp_frame_size;
    _nframes          = req.tp_frame_nr;

    printf("Block Size: \t\t%d\nNumber of Blocks: \t%d\nFrame Size: \t\t%d\nNumber of frames: \t%d\n",
            req.tp_block_size, req.tp_block_nr, req.tp_frame_size, req.tp_frame_nr);

    if(setsockopt(_socket, SOL_PACKET, PACKET_RX_RING, (void *) &req, sizeof(req)) < 0)
    {
        perror("setsockopt()");
        close(_socket);
        exit(-1);
    }

    // Map kernel buffer to user space using mmap
    _map = (char *) mmap(NULL, req.tp_block_nr * req.tp_block_size, PROT_READ | PROT_WRITE, MAP_SHARED, _socket, 0);
    if (_map == MAP_FAILED)
    {
        perror("mmap()");
        close(_socket);
        exit(-1);
    }

    // Allocate and initialise the ring buffer
    _ring = (struct iovec *) malloc(req.tp_frame_nr * sizeof(struct iovec));
    for(unsigned i = 0; i < req.tp_frame_nr; i++)
    {
        _ring[i].iov_base = _map + i * req.tp_frame_size;
        _ring[i].iov_len  = req.tp_frame_size;
    }
    
    // We are ready to start receiving data...
    
}

// Set double buffer
void PacketChunker::setDoubleBuffer(DoubleBuffer *buffer)
{
    _buffer = buffer;
    _heap = _buffer -> setHeapParameters(_nchans, _nsamp);
}    

// Run the UDP receiving thread
void PacketChunker::run()
{
    // Define variables for packet augmentation
    unsigned long _currTime = 0;
    unsigned _numPackets = 0;

    // Main processing loop
    unsigned long global_counter = 0;
    for(unsigned i = 0;;)
    {
        // Fetch next frame and check whether it is available for processing
        struct tpacket_hdr *header = (struct tpacket_hdr *) _ring[i].iov_base;

        // Data not available yet, wait until it is
        // We can just spin lock over here, packets will be available very soon
        while(!(header -> tp_status & TP_STATUS_USER))
        {
          //  if (global_counter % 1000 == 0)
          //      printf("Busy spin [%ld - %i]\n", global_counter, i);
            ; // Busy-spin
        }

        // Data is now available, we can process the current frame

        // Sanity check for frame validty
        if (header -> tp_status & TP_STATUS_COPY)
            fprintf(stderr, "Skipped packet: incomplete\n");

        // NOTE: The data routines will be able to detect whether we're losing packets

        // Get pointer to frame data
        unsigned char *frame      = (unsigned char *) header + FRAME_OFFSET; 

        // Extract IP information from packet
        struct iphdr  *ip_header  = (struct iphdr  *) (frame + sizeof(ethhdr));

        // TODO: Check whether packet was meant for us

        // Extract UDP information
        struct udphdr *udp_header = (struct udphdr *) (((char *) ip_header) + ip_header -> ihl * 4);

        // Check whether ports match, otherwise skip
        if (ntohs(udp_header -> dest) != _port)
        {
            printf("This packet was not meant for us [%d], %d\n", ntohs(udp_header -> dest), _port);
            header -> tp_status = 0;
            i = ( i == _nframes - 1) ? 0 : i + 1;
            continue;
        }

        global_counter++;

        // Get UDP packet contents
        unsigned char *data = (unsigned char *) (((char *) udp_header) + sizeof(udphdr));

        // Process packet header
        long unsigned int data_header =  be64toh(((uint64_t *) data)[0]);
        unsigned long  time    = data_header >> 26;
        unsigned short antenna = (data_header & 0x000000000000FFFF) * 2;
        unsigned short channel = (data_header >> 16) & 0x03FF;

        // Check if we are processing a new heap
        if (_currTime == 0) _currTime = time;

        // Check if the time in the header corresponds to the time of the
        // heap being processed, or if we've advanced one heap
        if (_currTime != time)
        {
            // We have received a packet from the previous buffer
            if (time < _currTime)
                fprintf(stderr, "PacketChunker: Received out of place packer, discarding\n");

            // We have moved on to a new heap (lost some packets)
            else
            {
                // We are processing a packet from a new heap
                printf("We have lost some packets: %d of %d for %ld\n", 
                        _numPackets, _npackets, _currTime);

                // Forward heap to Double Buffer (with timing parameters)
                // This will return a new heap 
//                _heap =  _buffer -> writeHeap(1351174098.5 + (1024 * _currTime) / (40e6/2.0/128.0),
//                                              1 / 19531.25);

                // Copy new packet to new heap
  //              memcpy(_heap + channel * _nantennas * _nsamp + antenna * _nsamp, &(data[4]), PACKET_DATA_LEN);

                _numPackets = 0;
                _currTime = time;
            }
        }

        // Everything OK, process
        else
        {
            // Place packet data in circular buffer
       //     memcpy(_heap + channel * _nantennas * _nsamp + antenna * _nsamp, &(data[4]), PACKET_DATA_LEN);

       // if (abs((int) (_numPackets - _npackets)) < 1000)
          //   printf("Read packet [%d - %d]\n", _numPackets, _npackets);

            // Increment packet number
            _numPackets++;

            // If we have processed enough packet to form one heap, advance to the next heap
            if (_numPackets == _npackets)
            {
                printf("Oh yeah, full heap\n");
                // Forward heap to Double Buffer (with timing parameters)
                // This will return a new heap pointer
            //    _heap =  _buffer -> writeHeap(1351174098.5 + (1024 * _currTime) / (40e6/2.0/128.0),
            //                                  1 / 19531.25);
                _currTime = 0;
                _numPackets = 0;
            }
        }

        // Tell kernel that this packet has been processed
        header -> tp_status = 0;

        // Advance to next frame
        i = ( i == _nframes - 1) ? 0 : i + 1;
    }
}
