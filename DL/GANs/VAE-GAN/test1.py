import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)
plt.plot(x, np.sin(x),'r-o')
plt.plot(x, np.sin(2 * x),'g--')
plt.plot(x, np.sin(3 * x))
plt.show()