import numpy as np
import itertools

# set k and make the example set
k = 2
s1 = [0, 1, 2]
s2 = [.1, 1.1, 1.9]

# %%
# create the distance matrix
newarray = [[abs(s2j - s1i) for s2j in s2] for s1i in s1]
distmat = np.array(newarray)

# %%

# get the nearest neighbors for each set
neighbors_si = np.argsort(distmat)
neighbors_sj = np.argsort(distmat.T)

# %%

# map element of each set to k nearest neighbors
neighbors_si = {i: neighbors_si[i][0:k] for i in range(len(neighbors_si))}
neighbors_sj = {j: neighbors_sj[j][0:k] for j in range(len(neighbors_sj))}

# %%

# for each combination of i and j determine if they are in each others neighbor list
for i, j in itertools.product(neighbors_si.keys(), neighbors_sj.keys()):
    if j in neighbors_si[i] and i in neighbors_sj[j]:
        print('{} and {} are {}-reciprocals'.format(s1[i], s2[j], k))

