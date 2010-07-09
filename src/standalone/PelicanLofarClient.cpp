// MDSM stuff
#include "PelicanLofarClient.h"

// C++ stuff
#include <iostream>
#include <complex>
#include "math.h"

// Connect to the Pelican-Lofar pipline and register for data acquisition
PelicanLofarClient::PelicanLofarClient(QString blobType, QString server, unsigned port)
{
    // Create the PelicanBlobClient instance, which will connect to the
    // TCPBlobServer and register this client to receive the requested blob type
    client = new PelicanBlobClient(blobType, server, port);
    dataHash.insert("ChannelisedStreamData", &blob);

    //TODO: Set TCP socket buffer to a large enough size to store packets
    //      between getNextBuffer calls
}

// Fills in input_buffer with nsamp samples by reading data off pelican-lofar
// ASSUMPTION: requested nchans == expected nchans
// NOTE: this will block until all data is available...
int PelicanLofarClient::getNextBuffer(float *input_buffer, unsigned int nsamp)
{
    unsigned int i, samples = 0;

    // We need to repeatedly get data until we fill up the buffer
    while (samples < nsamp) {

        // Get next data blob
        client -> getData(dataHash);

        // Copy blob data to input_buffer (and get total power)
        std::complex<double> *data = blob.data();
        for(i = 0; i < blob.size(); i++) {
            input_buffer[samples * blob.size() + i] = sqrt(pow(data -> real(),2) + pow(data -> imag(),2));
            data++;
        }
        // memcpy(input_buffer + samples, blob -> _data, sizeof(std::complex<double>) * blob -> size());
 
        // Update number of samples read
        if (samples % 1000 == 0)
            std::cout << "Received blob: " << samples << "\tneed: " << nsamp << std::endl;
        samples++;
    }

    std::cout << "Filled up buffer" << std::endl;
    return samples;
}
