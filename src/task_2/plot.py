import pandas as pd
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt

w = np.array([-0.0236806, 4.98004])
test_data = np.arange(-10, 150, 1)
test_data = test_data.reshape(len(test_data), 1)
test_data = np.hstack((test_data, np.ones((1, len(test_data))).reshape((len(test_data), 1))))
pred_test =  np.exp(test_data.dot(w))

plt.plot(test_data[:,0], pred_test, c='red', label='func')

plt.xlabel('T')
plt.ylabel('Z')

plt.legend()
plt.grid()
plt.show()