from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import numpy as np


def compute_distance(point, dataset):
    """ Compute the distance vector between on point and the entire dataset """

    return np.abs(dataset - point)    

def dbscan(values, min_number, epsilon = []):
    m, n = np.shape(values)
    
    if epsilon == []:
        # Calculate epsilon automatically, not really required
        pass

    # Initialise containers
    cluster_type   = np.zeros(m)
    touched        = np.zeros(m)
    cluster_class  = np.zeros(m)
    index_range    = np.arange(0, m)
    cluster_number = 1

    # Loop over all data points
    for i in range(m):

        # Check if data point already belongs to a cluster
        if touched[i] == 0:

            point    = np.array(values[i,:])[0]
            distance = compute_distance(point, values)

            # Find all points which fall within a certain range of the data point                    
            indices = -np.sign(distance - epsilon)
            indices[indices >= 0] = 1                         
            indices[indices < 0] = 0     
            indices = np.alltrue(indices, axis=1)  
            indices = index_range[indices.view(np.ndarray)[:,0]]                   

            # If not enough neighbours are found, then this is a solitary point
            if len(indices) > 1 and len(indices) < min_number + 1:
                cluster_type[i]  = 0
                cluster_class[i] = 0

            # Only this point belongs to the cluster
            if len(indices) == 1:
                cluster_type[i]  = -1
                cluster_class[i] = -1
                touched[i]       = 1

            # Enough indices to form a cluster
            if len(indices) >= min_number + 1:
                cluster_type[0] = 1;
                for val in indices: cluster_class[val] = cluster_number

                # Keep looping while we still have datapoints in the indices array
                while indices != []:
                    point = np.array(values[indices[0],:])[0]
                    touched[indices[0]] = 1
                    top_index = indices[0]
                    indices = np.delete(indices, 0)
                    
                    # Find all points which fall within a certain range of the current data point
                    distance = compute_distance(point, values)  

                    new_indices = -np.sign(distance - epsilon)
                    new_indices[new_indices >= 0] = 1                         
                    new_indices[new_indices < 0] = 0     
                    new_indices = np.alltrue(new_indices, axis=1)  
                    new_indices = index_range[new_indices.view(np.ndarray)[:,0]]

                    if len(new_indices) > 1:
                        for val in new_indices: cluster_class[val] = cluster_number
                        if len(new_indices) >= min_number + 1:
                            cluster_type[top_index] = 1
                        else:
                            cluster_type[top_index] = 0

                        for i in range(len(new_indices)):
                            if touched[new_indices[i]] == 0:
                                touched[new_indices[i]] = 1
                                indices = np.append(indices, new_indices[i])
                                cluster_class[new_indices[i]] = cluster_number

                cluster_number = cluster_number + 1

    for i, val in enumerate(cluster_class):
        if val == 0:
            cluster_class[i] = -1
            cluster_type[i]  = -1

    return np.array(cluster_class), np.array(cluster_type)

if __name__ == "__main__":
    
    # Initialise parameters
    tsamp = 0.0000512;
    nsamp = 32768.0 * 2;
    
    # Load file
    f = open('/data/Data/Medicina/B0329+54_Nov_08/beam1_output.dat', 'r')
    data = f.read()
    data = [[float(value) for value in item.split(',') if value != ""] for item in data.split("\n")]
    data = np.matrix(data[:-1])
    f.close()
    
    # File containing cluster information
    f = open('clusters_beam1.dat', 'w')
    
    start = 0
    index = 0
    clusters = 0
    # Loop through all the file
    while True:
        # Keep looping until we reach the end of the current buffer
        while True:
            # Check if point belongs to current data buffer
            if data[index,0] < data[start,0] + nsamp * tsamp:
                index += 1
                if index == np.shape(data)[0]: break
            else:
                break
        
        # Perform clustering and update index
        cluster_class, cluster_type = dbscan(data[start:index-1], 20, np.array([0.01, 10, 10]))    
        
        cluster_class[cluster_class != -1] += clusters
        clusters += len(np.unique(cluster_class)) - 1 if -1 in cluster_class else len(np.unique(cluster_class))
        
        
        # Output cluster file containing data points and cluster information
        for i, item in enumerate(data[start:index-1]):
            f.write("%f,%f,%f,%d,%d\n" % (item[0,0], item[0,1], item[0,2], cluster_class[i], cluster_type[i]))
        
        if index == np.shape(data)[0]: break    
        
        start = index
    
    fig = plt.figure()
    ax = Axes3D(fig)

    colours = ['r', 'g', 'b', 'k', 'y']
    print "Clusters: ", np.unique(cluster_class)
    for i in np.unique(cluster_class):
        ax.scatter(data[cluster_class == i,0], data[cluster_class == i, 1], data[cluster_class == i, 2], c = colours[int(i % len(colours))], marker='o')
    plt.show()
  
