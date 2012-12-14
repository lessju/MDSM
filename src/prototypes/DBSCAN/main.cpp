#include "vector"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "iostream"
#include "QTime"

using namespace std;

// DataPoint structure
typedef struct DataPoint
{
    double time, dm, snr;
    int   cluster, type;
};

FILE *hist_output = fopen("hist_output.dat", "wb");

// ------------------------- Cluster class -----------------------------------
class Cluster
{
public:

    // Class constructor
    Cluster(unsigned id, const vector<DataPoint> &dataPoints)
    {
        this -> dataPoints = dataPoints;
        this -> id = id;
    }

    // Add point to current cluster
    void addPoint(unsigned pointIndex)
    {
        indices.push_back(pointIndex);
    }

    // Compute a value indicating the probability that this cluster
    // is due to an astrophysical transient
    float computeTransientProbability(float minDm, float dmStep, int numDMs)
    {
        // Initialise DM-SNR histogram
        float dmHistogram[numDMs];
        memset(dmHistogram, 0, numDMs * sizeof(float));

        // Generate DM-SNR curve for current cluster (flatten in the X-dimenion)
        for(unsigned i = 0; i < indices.size(); i++)
        {
            DataPoint point = dataPoints[indices[i]];
            dmHistogram[(int) round((point.dm - minDm) / dmStep)] += point.snr;
        }

        fwrite(dmHistogram, sizeof(float), numDMs, hist_output);
    }

    // Get cluster Id
    unsigned ClusterId() { return id; }

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
    DBScan(float minTime, float minDm, float minSnr, unsigned minPoints)
    {
        // Initialise variables
        this -> min_dm     = minDm;
        this -> min_snr    = minSnr;
        this -> min_time   = minTime;
        this -> min_points = minPoints;
    }

    // Get number of clusters
    int getNumberOfClusters() { return this -> numClusters; }

    // Perform clustering
    vector<Cluster*> performClustering(vector<DataPoint> &dataPoints)
    {
        // Initliase clustering
        char visited[dataPoints.size()];
        numClusters = 0;
        clusters.clear();

        // Loop over all data point
        for(unsigned i = 0; i < dataPoints.size(); i++)
        {
            // Check if data point has already been processed
            if (visited[i])
                continue;

            // Set as visitied
            visited[i] = 1;

            // Get list of neghbours
            vector<int> neighbors = getNeighbours(dataPoints, i);

            // if not enough points to create a cluster mark as noise
            if (neighbors.size() < min_points)
            {
                dataPoints[i].cluster = -1;
                continue;
            }

            Cluster *cluster = new Cluster(clusters.size() + 1, dataPoints);
            clusters.push_back(cluster);

            // Assign point to new cluster
            dataPoints[i].cluster = cluster -> ClusterId();
            cluster -> addPoint(i);

            // Loop over all neighbouring data points
            for(unsigned j = 0; j < neighbors.size(); j++)
            {
                // Check if data point has already been processed
                if (!visited[neighbors[j]])
                {
                    // Mark data point as visited
                    visited[neighbors[j]] = 1;

                    // Get list of neighbours for current point
                    vector<int> current_neighbors = getNeighbours(dataPoints, neighbors[j]);

                    // Check if there are any neighbors
                    if (current_neighbors.size() >= 1)
                    {
                        for(unsigned k = 0; k < current_neighbors.size(); k++)
                            neighbors.push_back(current_neighbors[k]);
                    }
                }

                // If not already assigned to a cluster
                if (dataPoints[neighbors[j]].cluster == 0)
                {
                    dataPoints[neighbors[j]].cluster = cluster -> ClusterId();;
                    cluster -> addPoint(neighbors[j]);
                }
            }
        }

        this -> numClusters = clusters.size();

        return clusters;
    }


private:

    // Candidate selection for FDBSCAN
    vector<int> selectCandidates(const vector<DataPoint> &dataPoints, vector<int> &neighbours)
    {
        // Initialise limits
        unsigned min_x = 0, max_x = 0, min_y = 0, max_y = 0, min_z = 0, max_z = 0;

        // Loop over all neighborhood
        for(unsigned i = 1; i < neighbours.size(); i++)
        {
            min_x = (dataPoints[min_x].time < dataPoints[neighbours[i]].time) ? min_x : i;
            max_x = (dataPoints[max_x].time > dataPoints[neighbours[i]].time) ? max_x : i;
            min_y = (dataPoints[min_y].dm < dataPoints[neighbours[i]].dm) ? min_y : i;
            max_y = (dataPoints[max_y].dm > dataPoints[neighbours[i]].dm) ? max_y : i;
            min_z = (dataPoints[min_z].snr < dataPoints[neighbours[i]].snr) ? min_z : i;
            max_z = (dataPoints[max_z].snr > dataPoints[neighbours[i]].snr) ? max_z : i;
        }

        int arr[] = {neighbours[min_x], neighbours[max_x],
                     neighbours[min_y], neighbours[max_y],
                     neighbours[min_z], neighbours[max_z]};

        vector<int> candidates(arr, arr + sizeof(arr) / sizeof(arr[0]) );

        return candidates;
    }

    // Construct list of neighbours
    vector<int> getNeighbours(const vector<DataPoint> &dataPoints, unsigned index)
    {
        vector<int> neighbours;

        // Loop over all data points
        for(unsigned i = 0; i < dataPoints.size(); i++)
        {
            // Avoid computing distance to same point
            if (index != i)
            {
                // Compute distance to current point
                float x = fabs(dataPoints[index].time - dataPoints[i].time);
                float y = fabs(dataPoints[index].dm   - dataPoints[i].dm);
                float z = fabs(dataPoints[index].snr  - dataPoints[i].snr);

                // Check if distance is within legal bounds
                if (x < min_time && y < min_dm && z < min_snr)
                    neighbours.push_back(i);
               }
        }

        return neighbours;
    }

    // List of clusters
    vector<Cluster*> clusters;

    int      min_points;
    float    min_dm, min_time, min_snr;
    int      numClusters;
};

int main()
{
    // Open file for reading
    FILE *fp = fopen("/data/Data/Medicina/B0329+54_Nov_08/B0329+54_beam_0_2012-11-07_23:22:49.dat", "r");
    //FILE *fp = fopen("/home/lessju/Code/MDSM/release/TestBeam_beam_0.dat", "r");

    // Parse input file
    vector<DataPoint> dataPoints;
    double time, dm, snr;
    int counter = 0;
    while (fscanf(fp, "%lf,%lf,%lf", &time, &dm, &snr) != EOF && counter < 4096*8)
    {
        // Create data point
        DataPoint point = {time, dm, snr, 0, 0};
        dataPoints.push_back(point);
        counter++;
    }
    fclose(fp);

    // Start timer
    QTime myTimer;
    myTimer.start();

    // Perofrm clustering
    DBScan clustering(0.005, 5, 5, 20);
    vector<Cluster*> clusters = clustering.performClustering(dataPoints);

    // Examine clusters
    for(unsigned i = 0; i < clusters.size(); i++)
    {
        clusters[i] -> computeTransientProbability(0, 0.1, 768);
    }

    std::cout << "Found " << clustering.getNumberOfClusters() << " clusters in " <<  myTimer.elapsed() << "ms" << std::endl;

    // Output clustering results
    fp = fopen("cluster_output.dat", "w");
    for(unsigned i = 0; i < dataPoints.size(); i++)
        fprintf(fp, "%f,%f,%f,%d,%d\n", dataPoints[i].time, dataPoints[i].dm, dataPoints[i].snr,
                                        dataPoints[i].cluster, dataPoints[i].type);
    fclose(fp);
}
