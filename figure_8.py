import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# 1 = fig-8, 2 = butterfly, 3 = ying-yang
if len(sys.argv) == 1:
    initial_condition = 1
else:
    initial_condition = int(sys.argv[1])

dt = 0.001
if len(sys.argv) < 2:
    periods = 1
else:
    periods = float(sys.argv[2])

m1, m2, m3 = 1, 1, 1
g = 1

if initial_condition == 2:
    # Initial conditions for butterfly periodic solution
    v1 = (0.30689, 0.12551)
    period = 6.2356
elif initial_condition == 3:
    # Ying-yang 2a
    # v1 = (0.51394, 0.30474)
    v1 = (0.416822, 0.330333)
    # period = 17.3284
    period = 55.789829
else:
    # Initial conditions for figure 8 periodic solution
    x1, x2 =  (-0.97000436, 0.24308753), (0, 0)
    v1, v2 = (0.4662036850, 0.4323657300), (-0.93240737, -0.86473146)
    x3, v3 = (-x1[0], -x1[1]), v1
    period = 6.3259

if initial_condition in (2,3):
    # Isosceles colinear configuration
    x1 = (-1, 0)
    x2 = (1, 0)
    x3 = (0, 0)
    v2 = v1
    v3 = (-2*v1[0], -2*v1[1])

x0 = np.hstack((x1, x2, x3, v1, v2, v3))
t_end = periods*period

l2 = np.linalg.norm

def dynamics(t, y):
    x1 = y[:2]/m1
    x2 = y[2:4]/m2
    x3 = y[4:6]/m3
    dy = np.zeros(len(y))
    dy[0:6] = y[6:12]
    dy[6:8] = -g*m2*(x1-x2)/l2(x1-x2)**3 - g*m3*(x1-x3)/l2(x1-x3)**3
    dy[8:10] = -g*m3*(x2-x3)/l2(x2-x3)**3 - g*m1*(x2-x1)/l2(x2-x1)**3
    dy[10:12] = -g*m1*(x3-x1)/l2(x3-x1)**3 - g*m2*(x3-x2)/l2(x3-x2)**3
    return dy


sol = solve_ivp(dynamics, (0, t_end), x0, max_step=dt, method="Radau")

# Print solver status
if sol.success:
    status = "succeeded"
else:
    status = "failed"
print(f'IVP solver {status}: {sol.message}')

axlim = np.max(sol.y[:6])

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
for i in range(3):
    ax1.plot(sol.y[i*2], sol.y[i*2+1])
ax1.set_xlabel("$q_1$")
ax1.set_ylabel("$q_2$")
ax1.set_ylim(-axlim, axlim)
ax1.set_xlim(-axlim, axlim)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for i in range(3):
    ax2.plot(sol.y[i*2], sol.y[i*2+1], sol.t)
ax2.set_xlabel("$q_1$")
ax2.set_ylabel("$q_2$")
ax2.set_zlabel("time, $t$")
ax2.set_ylim(-axlim, axlim)
ax2.set_xlim(-axlim, axlim)

plt.suptitle("Figure-8 trajectory")

plt.show()
