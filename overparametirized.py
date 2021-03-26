import pdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

# Set random seed to repeat experiment
np.random.seed(99)

# Shortcut for Euclidian norm
l2 = np.linalg.norm

# Gradient of squared p-norm
def spnorm_grad(x, p):
    if np.linalg.norm(x, ord=p) < 1e-16:
        return np.zeros(x.shape)
    else:
        return x * (abs(x)/np.linalg.norm(x, ord=p))**(p-2)

# Gradient of squared conjugate p-norm
def spnorm_conj_grad(x, p):
    q = p/(p-1)
    return spnorm_grad(x,q)

# Simulation parameters, one period is 6.3259
dt = 0.005
t_end = 100

# Integration algorithm: 0 = scipy, 1 = forward Euler
algo = 0

# Parameter estimator parameters
gamma = 3.5
kp = 5
kq = 5
t_d = 0
r_d = 0
norm_order = 1.05
t_open = t_end

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

def parameter_gradient_dynamics(xh, x, t):
    """
        Dynamics of parameter estimate gradient
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
    gf13_1 = gforce_1(qh1, qh3, m1*m3)
    gf23_1 = gforce_1(qh2, qh3, m2*m3)

    gf12_2 = gforce_2(qh1, qh2, m1*m2)
    gf13_2 = gforce_2(qh1, qh3, m1*m3)
    gf23_2 = gforce_2(qh2, qh3, m2*m3)
    
    gf12_3 = gforce_3(qh1, qh2, m1*m2)
    gf13_3 = gforce_3(qh1, qh3, m1*m3)
    gf23_3 = gforce_3(qh2, qh3, m2*m3)

    # Partial derivatives
    del_Y = np.zeros((12, 21))
    del_Y[:,:12] = np.array([[0, 0, 0, 0, 0, 0, 2*ph1[0], 0, 0, 4*ph1[0]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2*ph1[1], 0, 0, 4*ph1[1]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 2*ph2[0], 0, 0, 4*ph2[0]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 2*ph2[1], 0, 0, 4*ph2[1]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 2*ph3[0], 0, 0, 4*ph3[0]**3],
                             [0, 0, 0, 0, 0, 0, 0, 0, 2*ph3[1], 0, 0, 4*ph3[1]**3],
                             [-2*qh1[0], 0, 0, -4*qh1[0]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [-2*qh1[1], 0, 0, -4*qh1[1]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, -2*qh2[0], 0, 0, -4*qh2[0]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, -2*qh2[1], 0, 0, -4*qh2[1]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, -2*qh3[0], 0, 0, -4*qh3[0]**3, 0, 0, 0, 0, 0, 0],
                             [0, 0, -2*qh3[1], 0, 0, -4*qh3[1]**3, 0, 0, 0, 0, 0, 0]])

    del_Y[6:,12:] = np.array([[-gf12_1[0], -gf13_1[0], 0, -gf12_2[0], -gf13_2[0], 0, -gf12_3[0], -gf13_3[0], 0],
                              [-gf12_1[1], -gf13_1[1], 0, -gf12_2[1], -gf13_2[1], 0, -gf12_3[1], -gf13_3[1], 0],
                              [gf12_1[0], 0, -gf23_1[0], gf12_2[0], 0, -gf23_2[0], gf12_3[0], 0, -gf23_3[0]],
                              [gf12_1[1], 0, -gf23_1[1], gf12_2[1], 0, -gf23_2[1], gf12_3[1], 0, -gf23_3[1]],
                              [0, gf13_1[0], gf23_1[0], 0, gf13_2[0], gf23_2[0], 0, gf13_3[0], gf23_3[0]],
                              [0, gf13_1[1], gf23_1[1], 0, gf13_2[1], gf23_2[1], 0, gf13_3[1], gf23_3[1]]])

    # Time dependent learning rate
    r_l = gamma/(1+r*t)
    return -r_l*del_Y.T @ (xh-x)

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
    gf13_1 = gforce_1(qh1, qh3, m1*m3)
    gf23_1 = gforce_1(qh2, qh3, m2*m3)

    gf12_2 = gforce_2(qh1, qh2, m1*m2)
    gf13_2 = gforce_2(qh1, qh3, m1*m3)
    gf23_2 = gforce_2(qh2, qh3, m2*m3)
    
    gf12_3 = gforce_3(qh1, qh2, m1*m2)
    gf13_3 = gforce_3(qh1, qh3, m1*m3)
    gf23_3 = gforce_3(qh2, qh3, m2*m3)

    # Partial derivatives
    del_Y = np.zeros((12, 21))
    del_Y[:,:12] = np.array([[0, 0, 0, 0, 0, 0, 2*ph1[0], 0, 0, 4*ph1[0]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2*ph1[1], 0, 0, 4*ph1[1]**3, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 2*ph2[0], 0, 0, 4*ph2[0]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 2*ph2[1], 0, 0, 4*ph2[1]**3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 2*ph3[0], 0, 0, 4*ph3[0]**3],
                             [0, 0, 0, 0, 0, 0, 0, 0, 2*ph3[1], 0, 0, 4*ph3[1]**3],
                             [-2*qh1[0], 0, 0, -4*qh1[0]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [-2*qh1[1], 0, 0, -4*qh1[1]**3, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, -2*qh2[0], 0, 0, -4*qh2[0]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, -2*qh2[1], 0, 0, -4*qh2[1]**3, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, -2*qh3[0], 0, 0, -4*qh3[0]**3, 0, 0, 0, 0, 0, 0],
                             [0, 0, -2*qh3[1], 0, 0, -4*qh3[1]**3, 0, 0, 0, 0, 0, 0]])

    del_Y[6:,12:] = np.array([[-gf12_1[0], -gf13_1[0], 0, -gf12_2[0], -gf13_2[0], 0, -gf12_3[0], -gf13_3[0], 0],
                              [-gf12_1[1], -gf13_1[1], 0, -gf12_2[1], -gf13_2[1], 0, -gf12_3[1], -gf13_3[1], 0],
                              [gf12_1[0], 0, -gf23_1[0], gf12_2[0], 0, -gf23_2[0], gf12_3[0], 0, -gf23_3[0]],
                              [gf12_1[1], 0, -gf23_1[1], gf12_2[1], 0, -gf23_2[1], gf12_3[1], 0, -gf23_3[1]],
                              [0, gf13_1[0], gf23_1[0], 0, gf13_2[0], gf23_2[0], 0, gf13_3[0], gf23_3[0]],
                              [0, gf13_1[1], gf23_1[1], 0, gf13_2[1], gf23_2[1], 0, gf13_3[1], gf23_3[1]]])


    # Change in state estimate
    return del_Y @ ah + np.hstack((kq*(x[:6]-xh[:6]), kp*(x[6:]-xh[6:])))

# Gravitational force between particles
def gforce_1(x1, x2, m):
    return g*m*(x1-x2)/(l2(x1-x2)**3)

def gforce_2(x1, x2, m):
    return 2*g*m*(x1-x2)/(l2(x1-x2)**4)

def gforce_3(x1, x2, m):
    return 3*g*m*(x1-x2)/(l2(x1-x2)**5)

def hamiltonian(x):
    return (-g*m1*m2/l2(x[0:2]-x[2:4], axis=0) - g*m2*m3/l2(x[2:4]-x[4:6], axis=0) - g*m1*m2/l2(x[0:2]-x[4:6], axis=0)
            + l2(x[6:8], axis=0)/(2*m1) + l2(x[8:10], axis=0)/(2*m1) + l2(x[10:12], axis=0)/(2*m1))

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

# Initialize estimate of state, parameters and parameter gradient
xh = np.zeros(sol.y.shape)
ah = np.zeros((21, len(sol.t)))
g_ah = np.zeros((21, len(sol.t)))

xh[:,0] = y0
a = np.array((0, 0 ,0, 0, 0, 0, 1/(2*m1), 1/(2*m2), 1/(2*m3),
                    0, 0, 0, g*m1*m2, g*m2*m3, g*m1*m3, 0, 0, 0, 0, 0, 0))/2
# noise_gain = 0.1
# ah[:,0] = a + np.random.uniform(-noise_gain, noise_gain, 21)
# ah[:,0] = np.random.uniform(0.1, 1.9, 21)
ah[:,0] = np.zeros(21)
g_ah[:,0] = spnorm_grad(ah[:,0], norm_order)

r = 0
# Do parameter estimation and simulate estimated dynamics
for i in range(len(sol.t)-1):
    delta_t = sol.t[i+1] - sol.t[i]
    xh[:,i+1] = xh[:,i] + estimate_dynamics(xh[:,i], sol.y[:,i], ah[:,i])*delta_t
    g_ah[:,i+1] = g_ah[:,i] + parameter_gradient_dynamics(xh[:,i], sol.y[:,i], sol.t[i])*delta_t
    ah[:,i+1] = spnorm_conj_grad(g_ah[:,i+1], norm_order)
    
    # Open control loop and end adaption after t_open seconds
    if sol.t[i] > t_open:
        kp = 0
        kq = 0
        gamma = 0

    # Decrease learning rate after t_d
    if sol.t[i] > t_d:
        r = r_d

# Plot position trajectory
plt.figure()
for i in range(0,6):
    plt.plot(sol.t, sol.y[i], color=viridis(i/6))
    plt.plot(sol.t, xh[i], linestyle="--", color=viridis(i/6))
plt.title('Particle positions')
plt.xlabel("$t$")
plt.ylabel("$q$")

# Plot velocity trajectory
plt.figure()
for i in range(6,12):
    plt.plot(sol.t, sol.y[i], color=viridis((i-6)/6))
    plt.plot(sol.t, xh[i], linestyle="--", color=viridis((i-6)/6))
plt.title('Particle velocities')
plt.xlabel(r"$t$")
plt.ylabel(r"$p$")

# Plot parameter estimate trajectory
plt.figure()
for i in range(ah.shape[0]):
    plt.plot(sol.t, ah[i,:])
plt.title("Parameter estimate trajectory")
plt.xlabel("time, $t$")
plt.ylabel("$\hat{a}$")

# Plot histogram of final parameter values
plt.figure()
plt.hist(ah[:,-1], bins=20)
plt.title("Parameter distribution")
plt.xlabel("Parameter value")
plt.ylabel("Frequency")

# Plot tracking error
plt.figure()
xe = np.sum(np.abs(sol.y - xh), axis=0)
plt.plot(sol.t, xe)
plt.title("Tracking error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{x} - x \Vert_1$")

# Plot parameter estimate error
plt.figure()
ae = np.sum(np.abs(ah.T - a).T, axis=0)
plt.plot(sol.t, ae)
plt.title("Parameter estimation error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{a} - a \Vert_1$")

# Plot parameter penalization function
plt.figure()
plt.plot(sol.t, 0.5*np.linalg.norm(ah, ord=norm_order, axis=0)**2)
plt.title("Penalization function")
plt.xlabel("time, $t$")
plt.ylabel(r"$\frac{1}{2} \Vert \hat{a} \Vert_p^2$")

# Plot learning rate
plt.figure()
plt.plot(sol.t, gamma/(1+r_d*sol.t))
plt.title("Learning rate")
plt.xlabel("$t$")
plt.ylabel(r"$\gamma$")

# Plot Hamiltonian error
plt.figure()
he = hamiltonian(xh) - hamiltonian(sol.y)
plt.plot(sol.t, he)
plt.title("Hamiltonian error")
plt.xlabel("$t$")
plt.ylabel(r"$\hat{\mathcal{H}} - \mathcal{H}$")

plt.show()
