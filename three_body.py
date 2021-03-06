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
t_end = 20 #6.3259

# Integration algorithm: 0 = scipy, 1 = forward Euler
algo = 0

# Parameter estimator parameters
gamma = 3.5
kp = 5
kq = 5

# Physical constants
g = 1
m1, m2, m3 = 1, 1, 1

# Initial conditions for figure 8 periodic solution
x1, x2 =  (-0.97000436, 0.24308753), (0, 0)
v1, v2 = (0.4662036850, 0.4323657300), (-0.93240737, -0.86473146)
x3, v3 = (-x1[0], -x1[1]), v1
x0 = np.array((x1, x2, x3, v1, v2, v3)).flatten()

# Initialize estimator
xh0 = x0
ah0 = np.zeros(6)

# Initialize simulation state
X0 = np.hstack((x0, xh0, ah0))

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

# System dynamics
def system_dynamics(x):
    x1 = x[:2]/m1
    x2 = x[2:4]/m2
    x3 = x[4:6]/m3
    dx = np.zeros(len(x))
    dx[0:6] = x[6:12]
    dx[6:8] = -g*m2*(x1-x2)/l2(x1-x2)**3 - g*m3*(x1-x3)/l2(x1-x3)**3
    dx[8:10] = -g*m3*(x2-x3)/l2(x2-x3)**3 - g*m1*(x2-x1)/l2(x2-x1)**3
    dx[10:12] = -g*m1*(x3-x1)/l2(x3-x1)**3 - g*m2*(x3-x2)/l2(x3-x2)**3
    return dx

def parameter_dynamics(xh, x):
    """
        Dynamics of parameter estimate ah
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
    gf12 = gforce(qh1, qh2)
    gf13 = gforce(qh1, qh3)
    gf23 = gforce(qh2, qh3)

    # Partial derivatives of basis vector Y
    dY = np.zeros((12,6))

    dY[:6,:3] = -np.array([[2*ph1[0], 0, 0],
                           [2*ph1[1], 0, 0],
                           [0, 2*ph2[0], 0],
                           [0, 2*ph2[1], 0],
                           [0, 0, 2*ph3[0]],
                           [0, 0, 2*ph3[1]]])

    dY[6:,3:] = np.array([[gf12[0], 0, gf13[0]],
                          [gf12[1], 0, gf13[1]],
                          [-gf12[0], gf23[0], 0],
                          [-gf12[1], gf23[1], 0],
                          [0, -gf23[0], -gf13[0]],
                          [0, -gf23[1], -gf13[1]]])

    # Change of parameter estimate
    dah = gamma*(dY.T @ (xh-x))
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
    gf12 = gforce(qh1, qh2)
    gf13 = gforce(qh1, qh3)
    gf23 = gforce(qh2, qh3)

    # Partial derivatives
    del_q_Y = np.array([[gf12[0], 0, gf13[0]],
                        [gf12[1], 0, gf13[1]],
                        [-gf12[0], gf23[0], 0],
                        [-gf12[1], gf23[1], 0],
                        [0, -gf23[0], -gf13[0]],
                        [0, -gf23[1], -gf13[1]]])

    del_p_Y = np.array([[2*ph1[0], 0, 0],
                        [2*ph1[1], 0, 0],
                        [0, 2*ph2[0], 0],
                        [0, 2*ph2[1], 0],
                        [0, 0, 2*ph3[0]],
                        [0, 0, 2*ph3[1]]])

    # Change in estimates
    dph = -del_q_Y @ ah[3:] + kp*(x[6:]-xh[6:])
    dqh = del_p_Y @ ah[:3] + kq*(x[:6]-xh[:6])
    return np.hstack((dqh, dph))

def total_dynamics(t, X):
    dX = np.zeros(len(X))
    dX[:12] = system_dynamics(X[:12])
    dX[12:24] = estimate_dynamics(X[12:24], X[:12], X[24:])
    dX[24:] = parameter_dynamics(X[12:24], X[:12])
    return dX

def gforce(x1, x2):
    return g*m2*(x1-x2)/l2(x1-x2)**3

# Solve IVP
if algo == 0:
    sol = solve_ivp(total_dynamics, (0, t_end), X0, max_step=dt)
elif algo == 1:
    sol = forward_euler(total_dynamics, (0, t_end), X0)

# Print solver status
if sol.success:
    status = "succeeded"
else:
    status = "failed"
print(f'IVP solver {status}: {sol.message}')

# Extract states
x = sol.y[:12]
xh = sol.y[12:24]
ah = sol.y[24:]

# Plot position trajectory
plt.figure(1)
for i in range(0,6):
    plt.plot(sol.t, x[i], color=viridis(i/6))
    plt.plot(sol.t, xh[i], linestyle="--", color=viridis(i/6))
plt.title('Particle positions')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# Plot velocity trajectory
plt.figure(2)
for i in range(6,12):
    plt.plot(sol.t, x[i], color=viridis((i-6)/6))
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
xe = np.sum(x - xh, axis=0)
plt.plot(sol.t, xe)
plt.title("Tracking error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\tilde{x}$")

plt.figure(5)
a = np.array((1/(2*m1), 1/(2*m2), 1/(2*m3), g*m1*m2, g*m2*m3, g*m1*m3))
ae = np.sum((ah.T - a).T, axis=0)
plt.plot(sol.t, ae)
plt.title("Parameter estimation error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\tilde{a}$")

plt.show()
