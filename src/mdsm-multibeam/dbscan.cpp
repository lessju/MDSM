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

// Class destructor
DBScan::~DBScan()
{
    // Destroy all cluster instances
    for(unsigned i = 0; i < clusters.size(); i++)
        delete clusters[i];
}

// Perform clustering
vector<Cluster*> DBScan::performClustering(vector<DataPoint> &dataPoints)
{
    // Initliase clustering
    char neighbors[dataPoints.size()]; // Store neighor points
    char visited[dataPoints.size()];   // Store visited
    numClusters = 0;
    clusters.clear();

    // Get pointer to underlying data points array
    DataPoint *points = reinterpret_cast<DataPoint *>(&dataPoints[0]);

    // Set neighbors and visited to 0
    memset(neighbors, 0, dataPoints.size() * sizeof(char));
    memset(visited, 0, dataPoints.size() * sizeof(char));

    // Loop over all data point
    for(unsigned i = 0; i < dataPoints.size(); i++)
    {
        // Check if data point has already been processed
        if (visited[i])
            continue;

        // Set as visitied
        visited[i] = 1;

        // Get list of neghbours
        unsigned found = getNeighbours(points, dataPoints.size(), i, neighbors, visited);

        // if not enough points to create a cluster mark as noise
        if (found < min_points)
        {
            dataPoints[i].cluster = -1;

            // Not enough point in neighborhood, reset neighor list
            memset(neighbors, 0, dataPoints.size());
            continue;
        }

        Cluster *cluster = new Cluster(clusters.size() + 1, dataPoints);
        clusters.push_back(cluster);

        // Assign point to new cluster
        dataPoints[i].cluster = cluster -> ClusterId();
        cluster -> addPoint(i);

        // Process all the found point
        while (found != 0)
        {
            // Get index of next neighboring point
            unsigned j = 0;
            while (neighbors[j] == 0)
            {
                j++;
                if (j >= dataPoints.size()) j = 0;
            }

            // Check if data point has already been processed
            if (!visited[j])
            {
                // Mark data point as visited
                visited[j] = 1;

                // Add point's neighborhood to current list
                found += getNeighbours(points, dataPoints.size(), j, neighbors, visited);
            }

            // If not already assigned to a cluster
            if (dataPoints[j].cluster == 0)
            {
                dataPoints[j].cluster = cluster -> ClusterId();
                cluster -> addPoint(j);
            }

            // Mark this neighbor as processed
            neighbors[j] = 0;
            found--;
        }
    }

    this -> numClusters = clusters.size();

    return clusters;
}

// Perform optimised clustering (FDBSCAN)
vector<Cluster*> DBScan::performOptimisedClustering(vector<DataPoint> &dataPoints)
{
    // Initliase clustering
    numClusters = 0;
    clusters.clear();

    // Get pointer to underlying data points array
    DataPoint *points = reinterpret_cast<DataPoint *>(&dataPoints[0]);

    // Loop over all data points
    for(unsigned i = 0; i < dataPoints.size(); i++)
    {
        // Check if data point has already been processed
        if (dataPoints[i].cluster > 0)
            continue;

        // Get list of neghbours
        vector<int> neighbors = getNeighboursVector(points, dataPoints.size(), i);

        // if not enough points to create a cluster mark as noise
        if (neighbors.size() < min_points)
        {
            dataPoints[i].cluster = -1;
            continue;
        }

        // Create new cluster
        Cluster *cluster = new Cluster(clusters.size() + 1, dataPoints);
        clusters.push_back(cluster);

        // Assign point and neighborhood to new cluster
        dataPoints[i].cluster = cluster -> ClusterId();
        cluster -> addPoint(i);

        // Get representative seeds for this cluster
        vector<int> candidates = selectCandidates(points, neighbors);

        // Loop over all core point candidates
        while (candidates.size() != 0)
        {
            // Get current candidate
            unsigned candidate = candidates[0];

            // Get neighborhood for current candidate
            neighbors = getNeighboursVector(points, dataPoints.size(), candidate);

            // Check if this is a core point
            if (neighbors.size() >= min_points)
            {
                // Select representative seeds
                vector<int> current_reps = selectCandidates(points, neighbors);

                // Add valid representatives to global list
                for(unsigned j = 0; j < current_reps.size(); j++)
                    if (dataPoints[current_reps[j]].cluster == 0)
                        candidates.push_back(current_reps[j]);
            }

            // Set the core point's neighbour list to the current cluster
            for(unsigned j = 0; j < neighbors.size(); j++)
                if (dataPoints[neighbors[j]].cluster <= 0)
                {
                    dataPoints[neighbors[j]].cluster = cluster -> ClusterId();
                    cluster -> addPoint(neighbors[j]);
                }

            candidates.erase(candidates.begin());
        }
    }

    // TODO: Remove clusters containing a small number of points:
    //       - Simply ignore points belonging to these clusters?
    //       - Try to merge these clusters with neighboring ones, if any? (This will take up some processing time)

    this -> numClusters = clusters.size();

    return clusters;
}

    // Candidate selection for FDBSCAN
vector<int> DBScan::selectCandidates(DataPoint *dataPoints, const vector<int> &neighbours)
{
    // Initialise limits
    unsigned min_x = neighbours[0], max_x = neighbours[0], min_y = neighbours[0];
    unsigned max_y = neighbours[0], min_z = neighbours[0], max_z = neighbours[0];

    // Loop over all neighboring points and select representatives
    for(unsigned i = 1; i < neighbours.size(); i++)
    {
        min_x = (dataPoints[min_x].time < dataPoints[neighbours[i]].time) ? min_x : neighbours[i];
        max_x = (dataPoints[max_x].time > dataPoints[neighbours[i]].time) ? max_x : neighbours[i];
        min_y = (dataPoints[min_y].dm   < dataPoints[neighbours[i]].dm)   ? min_y : neighbours[i];
        max_y = (dataPoints[max_y].dm   > dataPoints[neighbours[i]].dm)   ? max_y : neighbours[i];
        min_z = (dataPoints[min_z].snr  < dataPoints[neighbours[i]].snr)  ? min_z : neighbours[i];
        max_z = (dataPoints[max_z].snr  > dataPoints[neighbours[i]].snr)  ? max_z : neighbours[i];
    }

    int arr[] = {min_x, max_x, min_y, max_y, min_z, max_z};

    vector<int> candidates(arr, arr + sizeof(arr) / sizeof(arr[0]) );

    return candidates;
}

// Construct list of neighbours
unsigned DBScan::getNeighbours(DataPoint* dataPoints, unsigned numberOfPoints, unsigned index, char *neighbors, char *visited)
{
    unsigned count = 0;

    // Loop over all data points
    for(unsigned i = 0; i < numberOfPoints; i++)
    {
        // Avoid computing distance to same point
        if (index != i)
        {
            // Compute distance to current point
            float x = fabs(dataPoints[index].time - dataPoints[i].time);
            float y = fabs(dataPoints[index].dm   - dataPoints[i].dm);
            float z = fabs(dataPoints[index].snr  - dataPoints[i].snr);

            // Check if distance is within legal bounds
            if (x < min_time && y < min_dm && z < min_snr && neighbors[i] == 0 && visited[i] == 0)
            {
                neighbors[i] = 1;
                count++;
            }
        }
    }

    return count;
}

// Return neighbor list in vector form
vector<int> DBScan::getNeighboursVector(DataPoint* dataPoints, unsigned numberOfPoints, unsigned index)
{
    vector<int> neighbours;

    // Loop over all data points
    for(unsigned i = 0; i < numberOfPoints; i++)
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

     
