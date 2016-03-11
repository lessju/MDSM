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

#define C 299792458.0

inline double D2R(double x)
{
    return x * M_PI / 180.0;
}

inline double R2D(double x)
{
    return x * 180.0 / M_PI;
}

// Array class function implementation
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

// ================= Helper functions ============================
// Get Julian Date
double getLocalSideralTime(Array *array)
{
    // Get Current Time
    time_t curr_time = time(0);
    struct tm *now = gmtime(&curr_time);

    // Adjust date to conform with following calculations
    now -> tm_year += 1900;  // 0 is AD 0
    now -> tm_mon += 1;      // 1 - 12 month range

//    printf("UTC: %4d/%2d/%2d %2d:%2d:%2d\n", now->tm_year, now->tm_mon, now->tm_mday, now->tm_hour,now->tm_min,now->tm_sec);

    // Check if January or February
    if (now -> tm_mon <= 2)
    {
        now -> tm_year -= 1;
        now -> tm_mon += 12;
    }

    // Compute julian date
    double hour = now->tm_hour + now->tm_min/60.0 + now->tm_sec/3600.0;
    double julian_date =  floorf( 365.25 * (now -> tm_year + 4716.0)) +
                          floorf( 30.6001 *( now -> tm_mon + 1.0)) + 2.0 -
                          floorf( now->tm_year / 100.0 ) +
                          floorf( floorf( now -> tm_year / 100.0 ) / 4.0 ) +
                          now -> tm_mday - 1524.5 +
                          (now -> tm_hour + now -> tm_min / 60.0 + now -> tm_sec / 3600.0) / 24.0 -
                          2451545.0;

    // Get Local Sidereal Time
    double lst = fmod(100.46 + (0.985647 * julian_date) + 15 * hour + array->getReferenceLong(), 360);
    lst = (lst < 0) ? lst + 360 : lst;
//    cout << "Julian Date: " << julian_date << " Sidereal time: " << lst << endl;

    return lst;
}

// Calculate phase-delay shifts for all antennas, frequencies
void calculate_shifts(SURVEY *survey, Array *array)
{
    // Get local sidereal time
    double lst = getLocalSideralTime(array);

    // Get array information
    double lat = D2R(array -> getReferenceLat());

    // Loop over each beam which needs to be generated
    for(unsigned b = 0; b < survey -> nbeams; b++)
    {
        BEAM beam = survey -> beams[b];
        double bandwidth = fabs(survey -> nchans * beam.foff) * 1e6;

        // Get beam pointing information
        double dec = D2R(beam.dec);
        double ra  = D2R(beam.ra);
        double ha  = D2R(beam.ha);

        // Compute hour angle
        // If RA is 0, then generate scanning beam (set HA to beam specified value)
        // Otherwise calculate required HA
        if (fabs(ra - 0) > 0.01) 
            ha = D2R(lst) - ra;

         // Compute azimuth and altitude
         double el, az;
         el = asin(sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(ha));
         az = acos((sin(dec) - sin(el) * sin(lat)) / (cos(el) * cos(lat))) ;
	     az = (az != az) ? M_PI : ((sin(ha) < 0) ? az : 2 * M_PI - az);

        // Compute trigonometric factor and antenna position unit
        double trig_factor_y = cos(el) * cos(az);
        double trig_factor_x = cos(el) * sin(az);
	    double trig_factor_z = sin(el);
        double unit_position = C / (array -> getCenterFrequency() * 1e9);

        // Compute antenna path differences
        double path_difference[array -> numberOfAntennas()];
        for(unsigned i = 0; i < array -> numberOfAntennas(); i++)
        {
            double diff_x = array -> getAntenna(i).getX() * unit_position * trig_factor_x;
            double diff_y = array -> getAntenna(i).getY() * unit_position * trig_factor_y;
            double diff_z = array -> getAntenna(i).getZ() * unit_position * trig_factor_z;
            path_difference[i] = diff_x + diff_y + diff_z;
        }

        // Loop over frequency channels
        for(unsigned i = 0; i < survey -> nchans; i++)
        {
            // Compute center channel frequency
            double frequency = (array -> getCenterFrequency()) * 1e9 + 
                                bandwidth / 2 - beam.foff * 1e6 * i - beam.foff / 2;
            double wavelength = C / frequency;

            // Loop over each antenna
            for(unsigned j = 0; j < array -> numberOfAntennas(); j++)
            {
                // Negative since we want to compensate for the path difference
                double phase = -2 * M_PI * path_difference[j] / wavelength; 
                unsigned index = i * survey -> nbeams * array -> numberOfAntennas() + 
                                 j * survey -> nbeams + b;

                {
                    (survey -> beam_shifts)[index].x = cos(phase);
                    (survey -> beam_shifts)[index].y = sin(phase);
                }
            }

        }
    }
}
