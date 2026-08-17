import matplotlib.pyplot as plt
import numpy as np
from source import activity_map

recorded_sample_neuron = np.load("simulation_data/3d_recorded_sample_neuron.npy")
position = np.load("simulation_data/3d_recorded_position.npy")
time = np.load("simulation_data/3d_recorded_time.npy")
recorded_population = np.load("simulation_data/3d_recorded_population.npy")


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Create coordinate grids for the 3D array
# ax.plot(position[:, 0], position[:, 1], zs=position[:, 2])
nbins = 50
s, (x, y, z) = activity_map(position, recorded_sample_neuron, nbins=nbins)
coords = np.where(s > 0.5 * np.max(s))
ax.scatter(
    coords[0] / nbins * (x[-1] - x[0]) + x[0],
    coords[1] / nbins * (y[-1] - y[0]) + y[0],
    coords[2] / nbins * (z[-1] - z[0]) + z[0],
    c=s[coords],
    cmap="viridis",
    s=10,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of Neural Activity")
# plt.colorbar(ax.collections[0], ax=ax, label='Activity Level')
plt.show()
