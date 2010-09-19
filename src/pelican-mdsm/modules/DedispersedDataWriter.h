#ifndef SIGPROCSTOKESWRITER_H
#define SIGPROCSTOKESWRITER_H

#include "pelican/output/AbstractOutputStream.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"
#include <QtCore/QDataStream>
#include <QtCore/QFile>

#include <fstream>
#include <vector>

using namespace pelican;

class DedispersedDataWriter: public AbstractOutputStream
{
    public:
		DedispersedDataWriter( const ConfigNode& config );
        ~DedispersedDataWriter();

    public:
        virtual void send(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void WriteString(std::ofstream *file, QString string);
        void WriteInt(std::ofstream *file, QString name, int value);
        void WriteFloat(std::ofstream *file, QString name, float value);
        void WriteDouble(std::ofstream *file, QString name, double value);
        void WriteLong(std::ofstream *file, QString name, long value);

    private:
        QString       		  		_filePrefix;
        std::vector<std::ofstream*> _files;
        std::vector<float> 		  	_dmValues;
        float         		  		_fch1, _foff, _tsamp, _refdm;
        int _nTotalSubbands;
        unsigned _nChannels, _nSubbands, _clock, _integration ;

};

PELICAN_DECLARE(AbstractOutputStream, DedispersedDataWriter)

#endif // SIGPROCSTOKESWRITER_H
