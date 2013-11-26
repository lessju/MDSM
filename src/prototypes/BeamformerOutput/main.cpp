#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sched.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <net/if.h>
#include <sys/poll.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>

// ##########################################################################
// http://lxr.linux.no/linux+v2.6.36/Documentation/networking/packet_mmap.txt
// ##########################################################################

// Fixed port number
#define PORT         10000

// Data packet length
#define PACKET_LEN   256+16

// Maximum size allocatable by kmalloc
#define SIZE_MAX     131072

// Pointer size
#define POINTER_SIZE sizeof(void *)

// Maximum region size allocated by __get_free_pages()
#define MAX_ORDER    11

// Packet offset within frame
#define PKT_OFFSET      (TPACKET_ALIGN(sizeof(struct tpacket_hdr)) + \
                         TPACKET_ALIGN(sizeof(struct sockaddr_ll)))

// Hepler function for system calls return checks
void __check(int retval, const char *text)
{
    if (retval < 0)
    {
        perror(text);
        exit(-1);
    }
}

// Main function
int main()
{
    // Set scheduling policy
    struct sched_param sp;
    sp.__sched_priority = 50;
    if(sched_setscheduler(getpid(), SCHED_RR, &sp))
        perror("sched_setscheduler");

    // Create socket
    int _socket;
    __check(_socket = socket(PF_PACKET, SOCK_DGRAM, htons(ETH_P_IP)), "socket()");

    // Copy the interface name to ifreq structure
    struct ifreq s_ifr;
    strncpy(s_ifr.ifr_name, "eth0", sizeof(s_ifr.ifr_name));

    // Get interface index
    __check(ioctl(_socket, SIOCGIFINDEX, &s_ifr), "ioctl");

    // Fill sockaddr_ll struct to prepare for binding
    struct sockaddr_ll address;
    address.sll_family   = AF_PACKET;
    address.sll_protocol = 0x08;
    address.sll_ifindex  = s_ifr.ifr_ifindex;
    address.sll_hatype   = 0;
    address.sll_pkttype  = 0;
    address.sll_halen    = 0;

    // bind socket to eth0
    __check(bind(_socket, (struct sockaddr *) &address, sizeof(struct sockaddr_ll)), "bind()");

    // Get page size (in bytes) to calculate tpacket_req parameters
    long page_size = sysconf(_SC_PAGESIZE);

    // Set up PACKET_MMAP capturing mode (hard-coded values for now)
    struct tpacket_req req;
    req.tp_block_size = page_size * 1024;
    req.tp_block_nr   = 1;
    req.tp_frame_size = 512;
    req.tp_frame_nr   = req.tp_block_nr * req.tp_block_size / req.tp_frame_size;

    printf("Block Size: \t\t%d\nNumber of Blocks: \t%d\nFrame Size: \t\t%d\nNumber of frames: \t%d\n",
            req.tp_block_size, req.tp_block_nr, req.tp_frame_size, req.tp_frame_nr);

    __check(setsockopt(_socket, SOL_PACKET, PACKET_RX_RING, (void *) &req, sizeof(req)), "setsockopt()");

    // Map kernel buffer to user space using mmap
    char *map = (char *) mmap(NULL, req.tp_block_nr * req.tp_block_size, PROT_READ | PROT_WRITE, MAP_SHARED, _socket, 0);
    if (map == MAP_FAILED)
    {
        perror("mmap()");
        close(_socket);
        exit(-1);
    }

    // Allocate and initialise the ring buffer
    struct iovec *ring = (struct iovec *) malloc(req.tp_frame_nr * sizeof(struct iovec));
    for(unsigned i = 0; i < req.tp_frame_nr; i++)
    {
        ring[i].iov_base = map + i * req.tp_frame_size;
        ring[i].iov_len  = req.tp_frame_size;
    }

    // Define variables for packet augmentation
    unsigned long _currTime = 0;
    unsigned _numPackets = 0;
    unsigned _npackets = 16384;

    // Create a buffer to store incoming data in its place (to test memcpy speed)
    char *heap = (char *) malloc(32 * 1024 * 128 * sizeof(char));

    // Main processing loop
    unsigned long global_counter = 0;
    for(unsigned i = 0;;)
    {
        // Fetch next frame and check whether it is available for processing
        struct tpacket_hdr *header = (struct tpacket_hdr *) ring[i].iov_base;

        // Data not available yet, wait until it is
        // We can just spin lock over here, packets will be available very soon
        while(!(header -> tp_status & TP_STATUS_USER))
        {
            struct pollfd pfd;
            pfd.fd      = _socket;
            pfd.events  = POLLIN | POLLERR;
            pfd.revents = 0;
            poll(&pfd, 1, -1);
        }

//        global_counter++;
//        if (global_counter % 100000 == 0)
//        {
//                        printf("%ld\n", global_counter);
//        }


        // Data is now available, we can process the current frame

        // Sanity check for frame validty
        if (header -> tp_status & TP_STATUS_LOSING)
        {
            // Packet drops detected, get statistics from socket
//            struct tpacket_stats stats;
//            socklen_t size_sock = sizeof(tpacket_stats);
//            printf("We are losing packets\n");
//            if (getsockopt(_socket, SOL_PACKET, PACKET_STATISTICS, &stats, &size_sock) > -1)
//            {
//                printf("Dropped packets: [%d, %d, %ld]\n", stats.tp_drops, stats.tp_packets, header -> tp_status);
//            }
        }

        // NOTE: The data routines will be able to detect whether we're losing packets

        // Get pointer to frame data
        unsigned char *frame = (unsigned char *) header + PKT_OFFSET + 2; // Why +2, alignment?

        // Extract IP information from packet
        struct iphdr  *ip_header       = (struct iphdr  *) (frame + sizeof(ethhdr));

        // TODO: Check whether packet was meant for us

        // Extract UDP information
        struct udphdr *udp_header      = (struct udphdr *) (((char *) ip_header) + ip_header -> ihl * 4);

        // Check whether ports match, otherwise skip
        if (ntohs(udp_header -> dest) != PORT)
        {
            header -> tp_status = 0;
            i = ( i == req.tp_frame_nr - 1) ? 0 : i + 1;
            printf("Invalid packet\n");
            continue;
        }

        // Get UDP packet contents
        unsigned char *data = (unsigned char *) (((char *) udp_header) + sizeof(udphdr));

        long unsigned int data_header =  be64toh(((uint64_t *) data)[0]);
        unsigned long  time    = data_header >> 26;
//        unsigned short antenna = (data_header & 0x000000000000FFFF) * 2;
//        unsigned short channel = (data_header >> 16) & 0x03FF;

        // Check if we are processing a new heap
        if (_currTime == 0)
            _currTime = time;

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
                fprintf(stderr, "############### We have lost some packets: %d of %d for %ld\n", _numPackets, _npackets, _currTime);
                _numPackets = 0;
                _currTime = time;
            }
        }

        // Everything OK, process
        else
        {
            // Place packet data in circular buffer
            // memcpy(heap + channel * 32 * 128 + antenna * 128, &(data[4]), 256 );

            // If we have processed enough packet to form one heap, advance to the next heap
            _numPackets++;
            if (_numPackets == _npackets)
            {
                printf("Oh yeah full heap\n");
                _currTime = 0;
                _numPackets = 0;
            }
        }

        // Tell kernel that this packet has been processed
        header -> tp_status = 0;

        // Advance to next frame
        i = ( i == req.tp_frame_nr - 1) ? 0 : i + 1;
    }
}
