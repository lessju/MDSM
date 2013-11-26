#ifndef BEAMSHIFTS_H
#define BEAMSHIFTS_H

#include <QString>
#include <QList>

#include "survey.h"

// Antenna class : Assumes single polarisation
class Antenna
{
public:
    // Class constructor
    Antenna(unsigned id, QString name)
    {
        this -> id   = id;
        this -> name = name;
    }

    // Set antenna grid location
    void setGridLocation(unsigned x, unsigned y)
    {
        grid_x = x;
        grid_y = y;
    }

    // Set antenna position
    void setPosition(float x, float y, float z)
    {
        this -> x = x;
        this -> y = y;
        this -> z = z;
    }

    // Getters
    unsigned getId()    {return id; }
    QString getName()   {return name;}
    unsigned getGridX() { return grid_x; }
    unsigned getGridY() {return grid_y; }
    float getX() {return x; }
    float getY() {return y; }
    float getZ() {return z; }

private:
    unsigned id;
    QString name;
    unsigned grid_x, grid_y;
    float x, y, z;

};

// Array class
class Array
{
    public:

        // Default constructor
        Array() { }

        // Add antenna to array
        void addAntenna(Antenna antenna) { antennas.append(antenna); }

        unsigned numberOfAntennas() { return antennas.size(); }

        Antenna getAntenna(unsigned i) { return antennas[i]; }

        // Getters and Setters
        float getCenterFrequency() { return center_frequency; }
        void setCenterFrequency(float frequency) { center_frequency = frequency; }

        float getXSpacing() { return x_spacing; }
        void setXSpacing(float spacing) { x_spacing = spacing; }

        float getYSpacing() { return y_spacing; }
        void setYSpacing(float spacing) { y_spacing = spacing; }

        float getReferenceLat() { return reference_x; }
        void setReferenceLat(float x) {reference_x = x; }

        float getReferenceLong() { return reference_y; }
        void setReferenceLong(float y) {reference_y = y; }

private:

    QList<Antenna> antennas;

    float center_frequency; // MHz
    float x_spacing;        // m
    float y_spacing;        // m
    float reference_x;
    float reference_y;

};

// Declare functions
Array *processArrayFile(QString filepath);
void calculate_shifts(SURVEY *survey, Array *array);

#endif // BEAMSHIFTS_H
