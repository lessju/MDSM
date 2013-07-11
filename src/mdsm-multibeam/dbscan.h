#include "vector"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "iostream"
#include "QTime"
#include "survey.h"

using namespace std;

// DataPoint structure
typedef struct
{
    double time, dm, snr;
    int   cluster, type;
} DataPoint;

// ------------------------- Cluster class -----------------------------------
class Cluster
{
    public:

        // Class constructor
        Cluster(SURVEY *survey, unsigned id, const vector<DataPoint> &dataPoints);

        // Add point to current cluster
        void addPoint(unsigned pointIndex);

        // Compute a value indicating the probability that this cluster
        // is due to an astrophysical transient
        float computeTransientProbability(float minDm, float dmStep, unsigned numDMs);

        // Return number of points in cluster    
        unsigned numberOfPoints() { return indices.size(); }

        // Get cluster Id
        unsigned ClusterId() { return id; }

        // Get number of data points in cluster
        unsigned numPoints() { return indices.size(); }

        // Get maximum DM
        float getDM() { return this -> maxDM; }

        // Get SNR
        float getMaxSnr() { return this -> maxSNR; }

        // Get width
        float getWidth() { return this -> width; }

        // Get sample position
        double getPosition() { return this -> position; }

        // Return pointer to indices        
        vector<int> *getIndices() { return &indices; }

    private:
    
        // Store reference to data set
        vector<DataPoint> dataPoints;

        // List of point indices which make up the cluster
        vector<int> indices;

        // SURVEY pointer
        SURVEY *survey;

        // Cluster id
        unsigned id;

        // Store calculated cluster properties
        double position;
        float maxDM, maxSNR, width;
};

// ------------------------- DBSCAN ALGORITHM ---------------------------------
class DBScan
{
    public:
        // Class constructor
        DBScan(SURVEY *survey, float minTime, float minDm, float minSnr, unsigned minPoints);

        // Class destructor
        ~DBScan();

        // Get number of clusters
        int getNumberOfClusters() { return this -> numClusters; }

        // Perform clustering
        vector<Cluster*> performClustering(vector<DataPoint> &dataPoints);

        // Perform optimised clustering (FDBSCAN)
        vector<Cluster*> performOptimisedClustering(vector<DataPoint> &dataPoints);


    private:

        // Candidate selection for FDBSCAN
        vector<int> selectCandidates(DataPoint *dataPoints, const vector<int> &neighbours);

        // Construct list of neighbours
        unsigned getNeighbours(DataPoint* dataPoints, unsigned numberOfPoints, unsigned index, char *neighbors, char *visited);

        // Return neighbor list in vector form
        vector<int> getNeighboursVector(DataPoint* dataPoints, unsigned numberOfPoints, unsigned index);

        // List of clusters
        vector<Cluster*> clusters;

        // Survey pointer
        SURVEY *survey;

        // Other miscellaneous class variables
        unsigned min_points;
        float    min_dm, min_time, min_snr;
        int      numClusters;
};
