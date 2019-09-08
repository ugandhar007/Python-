import numpy as np

a=np.random.randint(low=1, high=20, size=15)

print("The Random 15 sized vector: ",a)

b = a.reshape((3,5))

print("Arranging the vector into 3*5 array: ",b)

d=np.max(b, axis=1)
print("Max element in each row: ",d)

c=np.where(b == np.amax(b,axis=1).reshape(-1,1), 0*b,b)
print("After replacing max values with zero: ",c)

