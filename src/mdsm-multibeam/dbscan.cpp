#include "dbscan.h"
#include "float.h"
#include "math.h"
#include "cpgplot.h"
#include <algorithm>

// ------------------------- CLUSTER CLASS ---------------------------------

// Class constructor
Cluster::Cluster(SURVEY *survey, unsigned id, const vector<DataPoint> &dataPoints)
{
    this -> dataPoints = dataPoints;
    this -> id = id;
    this -> survey = survey;
}

// Add point to current cluster
void Cluster::addPoint(unsigned pointIndex)
{
    indices.push_back(pointIndex);
}

// Compute a value indicating the probability that this cluster
// is due to an astrophysical transient
float Cluster::computeTransientProbability(float minDm, float dmStep, unsigned numDMs)
{
    // Initialise DM-SNR histogram
    float dmHistogram[numDMs];
    memset(dmHistogram, 0, numDMs * sizeof(float));

    // Generate DM-SNR curve for current cluster (flatten in the X-dimenion)
    for(unsigned i = 0; i < indices.size(); i++)
    {
        DataPoint pnt = (this -> dataPoints)[indices[i]];
        dmHistogram[(int) round((pnt.dm - minDm) / dmStep)] += pnt.snr;
    }

    // Smoothen DM-SNR curve using an N-element window moving average
    {
        float temp[numDMs];
        for(unsigned i = 1; i < numDMs - 1; i++)
            temp[i] = (dmHistogram[i] + dmHistogram[i] + dmHistogram[i + 1]) / 3.0;

        for(unsigned i = 1; i < numDMs - 1; i++)
            dmHistogram[i] = temp[i];
    }

    // Find all DMs having a normalised histogram value > 0.9, get the median
    // DM value and normalise by the mean value of the same range. This provides
    // better accuracy for wide pulses
    float maxValue = 0;
    {       
        for(unsigned i = 0; i < numDMs; i++) maxValue = max(maxValue, dmHistogram[i]);
        float invMaxValue = 1.0 / maxValue;

        vector<int> dmList;
        maxValue = 0;
        for(unsigned i = 0; i < numDMs; i++)
            if (dmHistogram[i] * invMaxValue > 0.9)
            {
                dmList.push_back(i);
                maxValue += dmHistogram[i];
            }

        unsigned size = dmList.size();

        // Compute mean high SNR
        maxValue /= (float) size;

        // Find median value
        std::sort(dmList.begin(), dmList.end());
        if (size % 2 == 0)
            this -> maxDM = (dmList[size / 2 - 1] + dmList[size / 2]) * 0.5 * dmStep + minDm;
        else
            this -> maxDM = dmList[size / 2] * dmStep + minDm;
    }

    // Sanity check on maximum DM
    if (this -> maxDM <= 2.0)
    {  
        printf("Cluster caused due to RFI (DM = %f)\n", this -> maxDM);
        return 1;
    }

    // Find maximum SNR for given DM
    this -> maxSNR = 0;
    for(unsigned i = 0; i < indices.size(); i++)
    {
        DataPoint pnt = (this -> dataPoints)[indices[i]];
        this -> maxSNR = (fabs(pnt.dm - maxDM) < dmStep * 0.5 && pnt.snr > this -> maxSNR) ? pnt.snr : this -> maxSNR;
    }
    
    // Find pulse FWHM
    double minT = FLT_MAX, maxT = FLT_MIN;
    for(unsigned i = 0; i < indices.size(); i++)
    {
        DataPoint pnt = (this -> dataPoints)[indices[i]];
        if (fabs(pnt.dm - this -> maxDM) < dmStep)
        {
            minT = (pnt.time < minT) ? pnt.time : minT;
            maxT = (pnt.time > maxT) ? pnt.time : maxT;
        }
    }
    this -> width = (maxT - minT) * 1e3 * 0.4;
    this -> position = minT + (maxT - minT) / 2;

    // Sanity checking on pulse width
    if (!(this -> width <= FLT_MAX && width >= FLT_MIN && this -> width > this -> survey -> min_pulse_width))
    {
        printf("Invalid cluster, pulse width is < %f, NaN or Inf = %f [%f, %f]\n", 
                this -> survey -> min_pulse_width, this -> width, maxT, minT);
        return 1;
    }

    printf("Cluster %d: maxDM = %f, maxSNR = %f,  minT = %f, maxT = %f, width = %f ms, ", 
            this -> id, this -> maxDM, this -> maxSNR, minT, maxT, this -> width);

    // Compute curve for incorrect de-disperison 
    // NOTE: Currently assumes homogeneous beams
    float snrFit[numDMs];
    float band = fabs((survey -> beams)[0].foff * survey -> nchans);
    float freq = pow(1e-3 * (survey -> beams)[0].fch1 - (band * 1e-3) * 0.5, 3);
    float fitMax = 0;
    for(unsigned i = 0; i < numDMs; i++)
    {
        // Calculate x-value 
        float x = i * dmStep + minDm - this -> maxDM;

        // Calculate y-value
        float y = 6.91e-3 * x * (band / (width * freq));
        snrFit[i] = sqrt(M_PI) * 0.5 * (1.0 / y) * erf(y);

        // Keep fit maximum for sanity check
        fitMax = (fitMax > snrFit[i]) ? fitMax : snrFit[i];
    }    
    
    // Overwrite snrFit value for MaxDM
    snrFit[(int) round((this -> maxDM - minDm) / dmStep)] = 1;

    // Sanity check on SNR fit
    if ((fitMax - 1.0) > 0.1)
    {
        printf("Invalid cluster, SNR fit not possible\n");
        return 1;
    }

    // Normalise SNR-DM histogram first
    for(unsigned i = 0; i < numDMs; i++) dmHistogram[i] /= maxValue;

    // Compute Mean Square Error between signals
    float mse = 0;
    for(unsigned i = 0; i < numDMs; i++)
        mse += (dmHistogram[i] - snrFit[i]) * (dmHistogram[i] - snrFit[i]);
    mse /= (float) numDMs;
   
    printf("MSE = %f\n", mse);

/*    if(cpgbeg(0, "/xwin", 1, 1) != 1) printf("Couldn't initialise PGPLOT\n");
    cpgask(false);
    float xvals[numDMs];
    for(unsigned i = 0; i < numDMs; i++) xvals[i] = i;
    cpgenv(0, numDMs, 0, 1, 0, 1);
    cpgsci(7);
    cpgline(numDMs, xvals, dmHistogram);
    cpgsci(8);
    cpgline(numDMs, xvals, snrFit);

    sleep(1.5);
*/
    return mse;
}

// ------------------------- DBSCAN CLASS ---------------------------------
// Class constructor
DBScan::DBScan(SURVEY *survey, float minTime, float minDm, float minSnr, unsigned minPoints)
{
    // Initialise variables
    this -> min_dm     = minDm;
    this -> min_snr    = minSnr;
    this -> min_time   = minTime;
    this -> min_points = minPoints;
    this -> survey     = survey;
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

        Cluster *cluster = new Cluster(this -> survey, clusters.size() + 1, dataPoints);
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
        Cluster *cluster = new Cluster(this -> survey, clusters.size() + 1, dataPoints);
        clusters.push_back(cluster);

        // Assign point and neighborhood to new cluster
        dataPoints[i].cluster = cluster -> ClusterId();
        cluster -> addPoint(i);

        for(unsigned j = 0; j < neighbors.size(); j++)
        {
            dataPoints[neighbors[j]].cluster = cluster -> ClusterId();
            cluster -> addPoint(neighbors[j]);
        }

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
        if (index != i && dataPoints[i].cluster <= 0)
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

     
