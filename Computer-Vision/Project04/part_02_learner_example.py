import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
# C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
# t_init = np.array([[-0.2, 0.3, -5.0]]).T
C_init = dcm_from_rpy([np.pi/1.3, -np.pi/8, -np.pi/6])
t_init = np.array([[-0.8, 0.8, -5.0]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

# Run experiments for different gain values
gains = [x/100.0 for x in range(2,202,2)]
steps = []
singulars = []
for gain in gains:
    num_steps, singular = ibvs_simulation(Twc_init, Twc_last, pts, K, gain, False)
    steps.append(num_steps)
    singulars.append(singular)
    print("gain:", gain, "num_steps:", num_steps)

plt.figure(figsize=(10, 6))

# Create plot
classes = ['Converged', 'Singularity']
colors = ListedColormap(['g','r'])
plt.rc('axes', axisbelow=True)
plt.grid(True)
scatter = plt.scatter(gains, steps, c=singulars, cmap=colors)
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.xlabel('Gain Value')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations Until Convergence')
plt.savefig('ibvs_gain_pt2.png')
plt.show()


# Sanity check the controller output if desired.
# ...

# gain = 0.1
# # Run simulation - use known depths.
# i = ibvs_simulation(Twc_init, Twc_last, pts, K, gain, False)
# print(i)