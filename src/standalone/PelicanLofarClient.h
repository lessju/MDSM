// Pelican stuff
#include "pelican/data/DataBlob.h"

// PelicanLofar stuff
#include "ChannelisedStreamData.h"
#include "PelicanBlobClient.h"

// Qt stuff
#include <QString>
#include <QHash>

using namespace pelican;
using namespace pelican::lofar;

class PelicanLofarClient 
{
    public:
        PelicanLofarClient(QString blobType, QString server, unsigned port);
        ~PelicanLofarClient() { }

    public:
        int getNextBuffer(float *input_buffer, unsigned int nsamp);       

    private:
        QHash<QString, DataBlob*> dataHash;
        ChannelisedStreamData blob;
        PelicanBlobClient *client;
};
