#include <QDomDocument>
#include <QStringList>
#include <QDomElement>
#include <QDomNode>
#include <QString>
#include <QFile>

#include <iostream>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "complex.h"

#include "beamshifts.h"
#include "survey.h"

using namespace std;

Array *processArrayFile(QString filepath)
{
    QDomDocument document("array");

    QFile file(filepath);
    if (!file.open(QFile::ReadOnly | QFile::Text))
        throw QString("Cannot open antenna file '%1'").arg(filepath);

    // Read the XML configuration file into the QDomDocument.
    QString error;
    int line, column;
    if (!document.setContent(&file, true, &error, &line, &column)) {
        throw QString("Config::read(): Parse error "
                "(Line: %1 Col: %2): %3.").arg(line).arg(column).arg(error);
    }

    QDomElement root = document.documentElement();
    if( root.tagName() != "array" )
        throw QString("Invalid root elemenent observation parameter xml file, should be 'array'");

    // Get the root element of the observation file
    QDomNode n = root.firstChild();

    // Initialise Array object
    Array *array = new Array();

    // Process Receiver tag
    while(!n.isNull())
    {
        if (QString::compare(n.nodeName(), QString("receiver"), Qt::CaseInsensitive) == 0)
        {
            n = n.firstChild();
            while(!n.isNull())
            {
                if (QString::compare(n.nodeName(), QString("center_freq"), Qt::CaseInsensitive) == 0)
                    array -> setCenterFrequency(n.firstChild().nodeValue().toFloat());
                n = n.nextSibling();
            }

        }
        n = n.nextSibling();
    }

    // Process grid tag
    n = root.firstChild();
    while(!n.isNull())
    {
        if (QString::compare(n.nodeName(), QString("grid"), Qt::CaseInsensitive) == 0)
        {
            n = n.firstChild();
            while(!n.isNull())
            {
                if (QString::compare(n.nodeName(), QString("x_spacing"), Qt::CaseInsensitive) == 0)
                    array -> setXSpacing(n.firstChild().nodeValue().toFloat());
                if (QString::compare(n.nodeName(), QString("y_spacing"), Qt::CaseInsensitive) == 0)
                    array -> setYSpacing(n.firstChild().nodeValue().toFloat());
                n = n.nextSibling();
            }

        }
        n = n.nextSibling();
    }

    // Process antennas tag
    n = root.firstChild();
    while(!n.isNull())
    {
        if (QString::compare(n.nodeName(), QString("antennas"), Qt::CaseInsensitive) == 0)
        {
            n = n.firstChild();
            while (!n.isNull())
            {
                // Check if reference
                if (QString::compare(n.nodeName(), QString("reference"), Qt::CaseInsensitive) == 0)
                {
                   QString lat = n.firstChild().firstChild().firstChild().nodeValue();
                   QString lon = n.firstChild().firstChild().nextSibling().firstChild().nodeValue();

                    // Calculate latitude in degress
                    float mult = (lat.contains("S")) ? -1 : 1;
                    QStringList values = lat.mid(0, lat.size() - 1).split(":");
                    float lat_degrees = 0;
                    for(int i = 0; i < values.size(); i++)
                        lat_degrees += values[i].toFloat() / (pow(60, i));

                    array -> setReferenceLat(mult * lat_degrees);

                    // Calculate longitude in degreess
                    mult = (lon.contains("W")) ? -1 : 1;
                    values = lon.mid(0, lon.size() - 1).split(":");
                    float lon_degrees = 0;
                    for(int i = 0; i < values.size(); i++)
                        lon_degrees += values[i].toFloat() / (pow(60, i));

                    array -> setReferenceLong(mult * lon_degrees);
                }


                // Otherwise, it should be an antenna
                if (QString::compare(n.nodeName(), QString("ant"), Qt::CaseInsensitive) == 0)
                {
                    unsigned id = n.toElement().attribute("id").toUInt();
                    QString name = n.toElement().attribute("name");
                    QDomNode e = n.firstChild().nextSibling();
                    float grid_x = e.firstChild().toElement().attribute("x").toFloat();
                    float grid_y = e.firstChild().toElement().attribute("y").toFloat();
                    float x = e.firstChild().nextSibling().firstChild().nodeValue().toFloat();
                    float y = e.firstChild().nextSibling().nextSibling(). firstChild().nodeValue().toFloat();
                    float z = e.firstChild().nextSibling().nextSibling().nextSibling().firstChild().nodeValue().toFloat();

                    Antenna antenna(id, name);
                    antenna.setGridLocation(grid_x, grid_y);
                    antenna.setPosition(x, y, z);
                    array->addAntenna(antenna);
                }

                n = n.nextSibling();
            }
        }
        n = n.nextSibling();
    }

    return array;
}

void calculate_shifts(SURVEY *survey, Array *array)
{
    // Loop over each beam which needs to be generated
    for(unsigned b = 0; b < survey -> nbeams; b++)
    {
        BEAM beam = survey -> beams[b];
        double bandwidth = fabs(survey -> nchans * beam.foff) * 1e6;

        // Compute source declination in degrees
        // If no source provided, use zenith (latitude of array)
        double dec = beam.dec * (M_PI / 180.0f);  // in radians

        // Compute zenith and pointing direction
        double zenith = array -> getReferenceLat() * M_PI / 180.0f;  // in radians
        double direction = zenith - dec;

        // Compute trigonometric factor and antenna position unit
        double trig_factor = sin(direction);
        double unit_position = 299792458.0 / (array -> getCenterFrequency() * 1e9);

        // Compute antenna path differences
        double path_difference[array -> numberOfAntennas()];
        for(unsigned i = 0; i < array -> numberOfAntennas(); i++)
            path_difference[i] = array -> getAntenna(i).getY() * unit_position * trig_factor;

        // Loop over frequency channels
        for(unsigned i = 0; i < survey -> nchans; i++)
        {
            // Compute center channel frequency
            double frequency = (array -> getCenterFrequency()) * 1e9 + bandwidth / 2 - beam.foff * 1e6 * i - beam.foff / 2;
            double wavelength = 299792458.0 / frequency;

            // Loop over each antenna
            for(unsigned j = 0; j < array -> numberOfAntennas(); j++)
            {
                double phase = -2 * M_PI * path_difference[j] / wavelength; // negative since we want to compensate for the path difference
                unsigned index = i * survey -> nbeams * array -> numberOfAntennas() + j * survey -> nbeams + b;
                (survey -> beam_shifts)[index].x = cos(phase);
                (survey -> beam_shifts)[index].y = sin(phase);
            }

        }
    }
}
