import numpy as np

myMatrix = np.loadtxt(open("ratings_data.txt", "r"), delimiter=" ", skiprows=0, dtype=int)
print(myMatrix)