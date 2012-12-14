#include "vector"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "iostream"
#include "QTime"

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
        Cluster(unsigned id, const vector<DataPoint> &dataPoints);

        // Add point to current cluster
        void addPoint(unsigned pointIndex);

        // Compute a value indicating the probability that this cluster
        // is due to an astrophysical transient
        float computeTransientProbability(float minDm, float dmStep, int numDMs);

        // Get cluster Id
        unsigned ClusterId() { return id; }

        // Get number of data points in cluster
        unsigned numPoints() { return indices.size(); }

        // Return pointer to indices        
        vector<int> *getIndices() { return &indices; }

    private:
        // Store reference to data set
        vector<DataPoint> dataPoints;

        // List of point indices which make up the cluster
        vector<int> indices;

        // Cluster id
        unsigned id;
};

// ------------------------- DBSCAN ALGORITHM ---------------------------------
class DBScan
{
    public:
        // Class constructor
        DBScan(float minTime, float minDm, float minSnr, unsigned minPoints);

        // Get number of clusters
        int getNumberOfClusters() { return this -> numClusters; }

        // Perform clustering
        vector<Cluster*> performClustering(vector<DataPoint> &dataPoints);


    private:

        // Candidate selection for FDBSCAN
        vector<int> selectCandidates(const vector<DataPoint> &dataPoints, vector<int> &neighbours);

        // Construct list of neighbours
        vector<int> getNeighbours(const vector<DataPoint> &dataPoints, unsigned index);

        // List of clusters
        vector<Cluster*> clusters;

        // Other miscellaneous class variables
        unsigned min_points;
        float    min_dm, min_time, min_snr;
        int      numClusters;
};
