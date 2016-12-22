import matplotlib.pyplot as plt
import numpy as np
import random

a = []
b = []
for i in range(10000):
    a.append(random.uniform(0,1))
    b.append(random.uniform(0,1))

plt.scatter(np.array(a),np.array(b))
plt.show()
