import numpy as np
import matplotlib.pyplot as plt

N = 100
ax = plt.subplot(111)
ax.set_ylim(32760,32780)
ax.plot(np.arange(N),np.random.randn(N)+32768)
plt.show()
