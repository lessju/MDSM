#ifndef SIGPROCWRITER_H
#define SIGPROCWRITER_H

#include "pelican/output/AbstractOutputStream.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"
#include <QDataStream>
#include <QFile>

#include <fstream>

using namespace pelican;
using namespace pelican::lofar;

class SigprocWriter : public AbstractOutputStream
{
    public:
        SigprocWriter( const ConfigNode& config );
        ~SigprocWriter();
        QString filepath() { return _filepath; }

    public:
        virtual void send(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void WriteString(QString string);
        void WriteInt(QString name, int value);
        void WriteFloat(QString name, float value);
        void WriteDouble(QString name, double value);
        void WriteLong(QString name, long value);

        // Data helpers

    private:
        QString       _filepath;
        std::ofstream _file;
        float         _fch1, _foff, _tsamp;
        int           _nchans;

};

PELICAN_DECLARE(AbstractOutputStream, SigprocWriter )

#endif // SIGPROCWRITER_H
