import numpy as np

x = np.arange(48).reshape((1,4,3,2,2))
print(x)

print(".............")
y = np.transpose(x, (0,2,1,3,4))
print(y)