
from numpy import array
from numpy.linalg import norm
from sklearn import preprocessing
a = array([1, 2, 3])
# print(a)

l2 = norm(a)
l2_norm= a / l2

print(sum(l2_norm))
print(l2_norm)
#%%
a = a.reshape(-1, 1)
result = preprocessing.normalize(a, norm='l2', axis=0) # axis = 0 along the column.
print(sum(result))