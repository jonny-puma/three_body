import pdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

# Shortcut for Euclidian norm
l2 = np.linalg.norm

# Simulation parameters
dt = 0.01
t_end = 50 #6.3259

# Integration algorithm: 0 = scipy, 1 = forward Euler
algo = 0

# Parameter estimator parameters
gamma = 0.5
kp = 2
kq = 8

# Physical constants
g = 1
m1, m2, m3 = 1, 1, 1

# Initial conditions for figure 8 periodic solution
x1, x2 =  (-0.97000436, 0.24308753), (0, 0)
v1, v2 = (0.4662036850, 0.4323657300), (-0.93240737, -0.86473146)
x3, v3 = (-x1[0], -x1[1]), v1
y0 = np.array((x1, x2, x3, v1, v2, v3)).flatten()

# System dynamics
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

# Simple forward Euler solver for debugging
def forward_euler(fun, t_span, y0):
    try:
        t = np.arange(t_span[0], t_span[1]+dt, dt) 
        l = len(t)
        y = np.zeros((len(y0), l))
        y[:,0] = y0
        for i in range(1,l):
            y[:,i] = y[:,i-1] + fun(t[i], y[:,i-1])*dt
        message = "Forward Euler successfull"
        success = True
    except Exception as e:
        message = f"Forward Euler failed. {e}"
        success = False
    
    return OdeResult(t=t, y=y, success=success, message=message)

def parameter_dynamics(xh, x):
    """
        Dynamics of parameter estimate a hat
        ph : p hat, estimate of p
        qh: q hat, estimagte of q
        pe: estimate error for p
        qe: estimate error for q
    """
    # Positions
    qh1 = xh[:2]
    qh2 = xh[2:4]
    qh3 = xh[4:6]

    # Velocities
    ph1 = xh[6:8]
    ph2 = xh[8:10]
    ph3 = xh[10:12]

    # Gravitational force vectors
    gf12_1 = gforce_1(qh1, qh2, m1*m2)
    gf13_1 = gforce_1(qh1, qh3, m1*m2)
    gf23_1 = gforce_1(qh2, qh3, m1*m2)

    gf12_2 = gforce_2(qh1, qh2, m1*m2)
    gf13_2 = gforce_2(qh1, qh3, m1*m3)
    gf23_2 = gforce_2(qh2, qh3, m2*m3)
    
    gf12_3 = gforce_3(qh1, qh2, m1*m2)
    gf13_3 = gforce_3(qh1, qh3, m1*m3)
    gf23_3 = gforce_3(qh2, qh3, m2*m3)

    # Partial derivatives
    del_Y = np.zeros((12, 21))
    del_Y[:,:12] = np.array([[2*qh1[0], 0, 0, 4*qh1[0]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [2*qh1[1], 0, 0, 4*qh1[1]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 2*qh2[0], 0, 0, 4*qh2[0]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, 2*qh2[1], 0, 0, 4*qh2[1]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2*qh3[0], 0, 0, 4*qh3[0]**3, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2*qh3[1], 0, 0, 4*qh3[1]**3, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, -2*ph1[0], 0, 0, -4*ph1[0]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, -2*ph1[1], 0, 0, -4*ph1[1]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, -2*ph2[0], 0, 0, -4*ph2[0]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, -2*ph2[1], 0, 0, -4*ph2[1]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, -2*ph3[0], 0, 0, -4*ph2[0]**3],
                             [0, 0, 0, 0, 0, 0, 0, 0, -2*ph3[1], 0, 0, -4*ph2[1]**3]])

    del_Y[:3,12:] = np.array([[-gf12_1[0], -gf13_1[0], 0, -gf12_2[0], -gf13_2[0], 0, -gf12_3[0], -gf13_3[0], 0],
                              [-gf12_1[1], -gf13_1[1], 0, -gf12_2[1], -gf13_2[1], 0, -gf12_3[1], -gf13_3[1], 0],
                              [gf12_1[0], 0, -gf23_1[0], gf12_2[0], 0, -gf23_2[0], gf12_3[0], 0, -gf23_3[0]],
                              [gf12_1[1], 0, -gf23_1[1], gf12_2[1], 0, -gf23_2[1], gf12_3[1], 0, -gf23_3[1]],
                              [0, gf13_1[0], gf23_1[0], 0, gf13_2[0], gf23_2[0], 0, gf13_3[0], gf23_3[0]],
                              [0, gf13_1[1], gf23_1[1], 0, gf13_2[1], gf23_2[1], 0, gf13_3[1], gf23_3[1]]])

    # Change in state estimate
    dah = gamma*0.5*del_Y.T @ np.flip(xh-x)
    return dah

def estimate_dynamics(xh, x, ah):
    """
        Dynamics of state estimate x hat
        ph: estimate og p
        qh: estimate of q
        ah: estimate of a
        p: measured p
        q: measured q
    """
    # Positions
    qh1 = xh[:2]
    qh2 = xh[2:4]
    qh3 = xh[4:6]

    # Velocities
    ph1 = xh[6:8]
    ph2 = xh[8:10]
    ph3 = xh[10:12]

    # Gravitational force vectors
    gf12_1 = gforce_1(qh1, qh2, m1*m2)
    gf13_1 = gforce_1(qh1, qh3, m1*m2)
    gf23_1 = gforce_1(qh2, qh3, m1*m2)

    gf12_2 = gforce_2(qh1, qh2, m1*m2)
    gf13_2 = gforce_2(qh1, qh3, m1*m3)
    gf23_2 = gforce_2(qh2, qh3, m2*m3)
    
    gf12_3 = gforce_3(qh1, qh2, m1*m2)
    gf13_3 = gforce_3(qh1, qh3, m1*m3)
    gf23_3 = gforce_3(qh2, qh3, m2*m3)

    # Partial derivatives
    del_Y = np.zeros((12, 21))
    del_Y[:,:12] = np.array([[-2*qh1[0], 0, 0, -4*qh1[0]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [-2*qh1[1], 0, 0, -4*qh1[1]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, -2*qh2[0], 0, 0, -4*qh2[0]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, -2*qh2[1], 0, 0, -4*qh2[1]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, -2*qh3[0], 0, 0, -4*qh3[0]**3, 0, 0, 0, 0, 0, 0],
                             [0, 0, -2*qh3[1], 0, 0, -4*qh3[1]**3, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2*ph1[0], 0, 0, 4*ph1[0]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2*ph1[1], 0, 0, 4*ph1[1]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 2*ph2[0], 0, 0, 4*ph2[0]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 2*ph2[1], 0, 0, 4*ph2[1]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 2*ph3[0], 0, 0, 4*ph2[0]**3],
                             [0, 0, 0, 0, 0, 0, 0, 0, 2*ph3[1], 0, 0, 4*ph2[1]**3]])

    del_Y[:3,12:] = -np.array([[-gf12_1[0], -gf13_1[0], 0, -gf12_2[0], -gf13_2[0], 0, -gf12_3[0], -gf13_3[0], 0],
                               [-gf12_1[1], -gf13_1[1], 0, -gf12_2[1], -gf13_2[1], 0, -gf12_3[1], -gf13_3[1], 0],
                               [gf12_1[0], 0, -gf23_1[0], gf12_2[0], 0, -gf23_2[0], gf12_3[0], 0, -gf23_3[0]],
                               [gf12_1[1], 0, -gf23_1[1], gf12_2[1], 0, -gf23_2[1], gf12_3[1], 0, -gf23_3[1]],
                               [0, gf13_1[0], gf23_1[0], 0, gf13_2[0], gf23_2[0], 0, gf13_3[0], gf23_3[0]],
                               [0, gf13_1[1], gf23_1[1], 0, gf13_2[1], gf23_2[1], 0, gf13_3[1], gf23_3[1]]])


    # Change in state estimate
    dxh = del_Y @ ah + np.hstack((kp*(x[6:]-xh[6:]), kq*(x[:6]-xh[:6])))
    return dxh

def gforce_1(x1, x2, m):
    return g*m*(x1-x2)/l2(x1-x2)**3

def gforce_2(x1, x2, m):
    return 2*g*m*(x1-x2)/l2(x1-x2)**4

def gforce_3(x1, x2, m):
    return 3*g*m*(x1-x2)/l2(x1-x2)**5

# Solve IVP
if algo == 0:
    sol = solve_ivp(dynamics, (0, t_end), y0, max_step=dt)
elif algo == 1:
    sol = forward_euler(dynamics, (0, t_end), y0)

# Print solver status
if sol.success:
    status = "succeeded"
else:
    status = "failed"
print(f'IVP solver {status}: {sol.message}')

# Initialize estimator
xh = np.zeros(sol.y.shape)
ah = np.zeros((21, len(sol.t)))

xh[:,0] = y0
ah[:,0] = np.random.uniform(0.1, 1.9, 6)   
# ah[:,0] = np.array((0.5, 0.5, 0.5, 1, 1, 1)) + np.random.uniform(-0.3, 0.3, 6)

# Do parameter estimation and simulate estimated dynamics
for i in range(len(sol.t)-1):
    del_t = sol.t[i+1] - sol.t[i]
    xh[:,i+1] = xh[:,i] + estimate_dynamics(xh[:,i], sol.y[:,i], ah[:,i])*del_t
    ah[:,i+1] = ah[:,i] + parameter_dynamics(xh[:,i], sol.y[:,i])*del_t

# Plot position trajectory
plt.figure(1)
for i in range(0,6):
    plt.plot(sol.t, sol.y[i], color=viridis(i/6))
    plt.plot(sol.t, xh[i], linestyle="--", color=viridis(i/6))
plt.title('Particle positions')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# Plot velocity trajectory
plt.figure(2)
for i in range(6,12):
    plt.plot(sol.t, sol.y[i], color=viridis((i-6)/6))
    plt.plot(sol.t, xh[i], linestyle="--", color=viridis((i-6)/6))
plt.title('Particle velocities')
plt.xlabel(r"$\dot{x}_1$")
plt.ylabel(r"$\dot{x}_2$")

# Plot paraeter estimate trajectory
plt.figure(3)
for i in range(ah.shape[0]):
    plt.plot(sol.t, ah[i,:])
plt.title("Parameter estimate trajectory")
plt.xlabel("time, $t$")
plt.ylabel("$\hat{a}$")

plt.figure(4)
xe = np.sum(np.abs(sol.y - xh), axis=0)
plt.plot(sol.t, xe)
plt.title("Tracking error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{x} - x \Vert_1$")

"""
plt.figure(5)
a = np.array((1/(2*m1), 1/(2*m2), 1/(2*m3), g*m1*m2, g*m2*m3, g*m1*m3))
ae = np.sum(np.abs(ah.T - a).T, axis=0)
plt.plot(sol.t, ae)
plt.title("Parameter estimation error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{a} - a \Vert_1$")
"""

plt.show()
