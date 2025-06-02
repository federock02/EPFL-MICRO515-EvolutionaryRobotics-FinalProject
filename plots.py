# plots fitness NSGA-II
""" import numpy as np
import matplotlib.pyplot as plt

PATH = "results/Ant_custom/multi/full_f.npy"

data = np.load(PATH)

generation, pop_size, opti_param = data.shape

avg_param = []

for i in range(generation):
    avg_param.append(np.mean(data[i], axis=0))

plt.plot(avg_param)
plt.title("Average fitness evolution per parameters")
plt.xlabel("Generation")
plt.ylabel("Average fitness")
plt.legend(['Forward speed', 'Control costs'])
plt.show() """

# plots fitness CMAES
import numpy as np
import matplotlib.pyplot as plt

PATH = "results/Ant_custom/single/full_f.npy"

data = np.load(PATH)

generation, pop_size = data.shape

avg_param = []

for i in range(generation):
    avg_param.append(np.mean(data[i], axis=0))

plt.plot(avg_param)
plt.title("Average fitness evolution")
plt.xlabel("Generation")
plt.ylabel("Average fitness")
plt.legend(['Forward speed'])
plt.show()