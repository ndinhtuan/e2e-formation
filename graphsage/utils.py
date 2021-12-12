import numpy as np

def create_all_edge(fformations, num_person=18):
    
    num_samples = int(num_person*(num_person-1)/2)
    all_edges = np.zeros([num_samples, 2], dtype=np.int16)
    all_labels = np.zeros([num_samples, 1], dtype=np.int16)

    c = 0
    for i in range(num_person):
        for j in range(i+1, num_person):
    
            all_edges[c] = [i, j]
            flag = False

            for formation in fformations:
                if i in formation and j in formation:
                    all_labels[c] = 1
                    flag = True
                    break

            if not flag:
                all_labels[c] = 0

            c += 1

    return all_edges, all_labels

def create_bin_edge(fformations, num_person = 18):
    r"""
    Create binary edge (array NxN) for training on SALSA dataset

    Arguments:
        ffromations: [ {list_of_ids_for_this_group}, {list_of_ids_for_this_group}, ... ]
    Returns:
        edge (Nx2 numpy array): edge array containing couple vertices
        bin_edge (NxN numpy array): binary edge array
    """
    
    num_samples = int(num_person*(num_person-1)/2)
    edges = []
    bin_edge = np.zeros([num_person, num_person], dtype=np.int16)
    
    c = 0
    for i in range(num_person):
        for j in range(i+1, num_person):

            for formation in fformations:
                if i in formation and j in formation:
                    bin_edge[i, j] = 1
                    edges.append([i, j])
                    break

            c += 1
    
    edges = np.array(edges, dtype=np.int16)
    return edges, bin_edge

def negative_sample(edges, num_samples, bin_adj_train):
    
    # Refer: https://github.dev/bwilder0/clusternet

    all_edges = np.zeros((edges.shape[0]*(num_samples+1), 2), dtype=np.int16)
    all_edges[:edges.shape[0]] = edges
    labels = np.zeros(all_edges.shape[0])
    labels[:edges.shape[0]] = 1
    idx = edges.shape[0]
    n = bin_adj_train.shape[0]
    for i in range(edges.shape[0]):
        for j in range(num_samples):
            #draw negative samples by randomly changing either the source or 
            #the destination node of this edge
#            idx = i*num_samples + j
            if np.random.rand() < 0.5: 
                to_replace = 0
            else: 
                to_replace = 1
            all_edges[idx, 1-to_replace] = edges[i, 1-to_replace]
            all_edges[idx, to_replace] = np.random.randint(0, n-1)
            while bin_adj_train[all_edges[idx, 0], all_edges[idx, 1]] == 1:
                all_edges[idx, to_replace] = np.random.randint(0, n-1)
            idx += 1
    return all_edges, labels

if __name__=="__main__":
    pass
