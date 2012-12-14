#include "dbscan.h"

// ------------------------- CLUSTER CLASS ---------------------------------

// Class constructor
Cluster::Cluster(unsigned id, const vector<DataPoint> &dataPoints)
{
    this -> dataPoints = dataPoints;
    this -> id = id;
}

// Add point to current cluster
void Cluster::addPoint(unsigned pointIndex)
{
    indices.push_back(pointIndex);
}

// Compute a value indicating the probability that this cluster
// is due to an astrophysical transient
float Cluster::computeTransientProbability(float minDm, float dmStep, int numDMs)
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

    return 0;
}

// ------------------------- DBSCAN CLASS ---------------------------------
// Class constructor
DBScan::DBScan(float minTime, float minDm, float minSnr, unsigned minPoints)
{
    // Initialise variables
    this -> min_dm     = minDm;
    this -> min_snr    = minSnr;
    this -> min_time   = minTime;
    this -> min_points = minPoints;
}

// Perform clustering
vector<Cluster*> DBScan::performClustering(vector<DataPoint> &dataPoints)
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

// Candidate selection for FDBSCAN
vector<int> DBScan::selectCandidates(const vector<DataPoint> &dataPoints, vector<int> &neighbours)
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
vector<int> DBScan::getNeighbours(const vector<DataPoint> &dataPoints, unsigned index)
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
