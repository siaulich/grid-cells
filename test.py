import numpy as np
import matplotlib.pyplot as plt
from grid_cells.random_walk import generate_bat_flight

T = 20
dt = 0.5e-3
T = 20
dt = 0.5e-3
bat_flight = generate_bat_flight(
    T=T,
    dt=dt,
    speed_correlation_time=5,
    initial_position=[4.5, 4.5, 2],
    initial_heading=[1, 1, 1],
)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

inverting = bat_flight["inverted"]

pos = bat_flight["pos"]
inverted = pos[inverting]
upright = pos[~inverting]


segments = np.stack([pos[:-1], pos[1:]], axis=1)
colors = np.where(inverting[:-1], "tab:orange", "tab:blue")

ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=1.2))
# Add legend entries for the trajectory color states
ax.plot([], [], color="tab:blue", lw=1.2, label="upright")
ax.plot([], [], color="tab:orange", lw=1.2, label="inverted")
ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="green", s=40, label="start")
ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="red", s=40, label="end")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("Bat trajectory")
ax.legend()
plt.tight_layout()
plt.show()
