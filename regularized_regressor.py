import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


# Set random seed for repeatability
np.random.seed(99)

# Shortcut for Euclidian norm
l2 = np.linalg.norm

# Gradient of squared p-norm
def spnorm_grad(x, p):
    # Define gradient as zero close to x=0 to avoid overflows
    if np.linalg.norm(x, ord=p) < 1e-16:
        return np.zeros(x.shape)
    else:
        return x * (abs(x)/np.linalg.norm(x, ord=p))**(p-2)

# Gradient of squared conjugate p-norm
def spnorm_conj_grad(x, p):
    if (p - 1) < 1e-6:
        q = np.inf
    else:
        q = p/(p-1)
    return spnorm_grad(x,q)

def bregman_divergence(norm_order, x1, x2):
    return l2(x1, ord=norm_order) - l2(x2, ord=norm_order) - spnorm_grad(x2, norm_order) @ (x1-x2)

def learning_rate(gamma, t, t_d, r_d, theta):
    if t < t_d:
        return gamma*((1-w_t) + w_t*np.cos(theta*t))
    elif t < t_end_learning:
        w_cos = w_t/(1+r_d*(t-t_d))
        return gamma*(1-w_cos + w_cos*np.cos(theta*t))/(1+r_d*(t-t_d))
    else:
        return 0

# Simulation parameters, one period is 6.3259
dt = 0.01
t_end = 100

# Integration algorithm: 0 = scipy, 1 = forward Euler
solver = 0

# First or second order adaption law
adaption_law = 1

# Parameter estimator parameters
gamma = 3.5
kp = 3
kq = 3
t_d = 40
t_end_learning = t_end
t_open = t_end
r_d = 1.5
theta = 0.3
w_t = 0
norm_order = 1.01

# Physical constants
g = 1
m1, m2, m3 = 1, 1, 1

# Initial conditions for figure 8 periodic solution
x1, x2 =  (-0.97000436, 0.24308753), (0, 0)
v1, v2 = (0.4662036850, 0.4323657300), (-0.93240737, -0.86473146)
x3, v3 = (-x1[0], -x1[1]), v1
x0 = np.array((x1, x2, x3, v1, v2, v3)).flatten()

# True parameters
"""
a = np.array((0, 0 ,0, 0, 0, 0, 1/(2*m1), 1/(2*m2), 1/(2*m3),
                    0, 0, 0, g*m1*m2, g*m2*m3, g*m1*m3, 0, 0, 0, 0, 0, 0))
"""
a = np.array((1/(2*m1), 1/(2*m2), 1/(2*m3), 0, 0 ,0, g*m1*m2, g*m2*m3, g*m1*m3, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0))/2

# Initial conditions for estimates
xh0 = x0
if norm_order >= 1 and norm_order <= 2:
    ah0 = np.zeros(21)
else:
    ah0 = 0.1*np.ones(21)
g_ah0 = spnorm_grad(ah0, norm_order)

# Initial condition for total simulation state
X0 = np.hstack((x0, xh0, g_ah0))

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

# Regression basis
def Y(x):
    Y = np.zeros(21)
    for i in range(3):
        p_i = x[i:i+2]
        q_i = x[i+6:i+8]
        Y[i] = q_i@q_i
        Y[i+3] = (q_i@q_i)**2
        Y[i+6] = p_i@p_i
        Y[i+9] = (p_i@p_i)**2
        
        j = (i+1)%3
        q_j = x[j:j+2]
        q_ij = 1/l2(q_i - q_j)
        Y[i+12] = q_ij
        Y[i+15] = q_ij**2
        Y[i+18] = q_ij**3
    return Y
    
# Regressor basis jacobian
def nabla_Y(x):
    # Positions
    qh1 = x[:2]
    qh2 = x[2:4]
    qh3 = x[4:6]

    # Velocities
    ph1 = x[6:8]
    ph2 = x[8:10]
    ph3 = x[10:12]

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

    # Partial derivative
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
    return del_Y

# True system dynamics
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

def parameter_gradient_dynamics_1(xh, x, t):
    """
        Dynamics of parameter estimate gradient
        ph : p hat, estimate of p
        qh: q hat, estimagte of q
        pe: estimate error for p
        qe: estimate error for q
    """
    # Regressor basis jacobian
    g_Y = nabla_Y(xh)

    # Time dependent learning rate
    r_l = learning_rate(gamma, t, t_d, r_d, theta)
    return -r_l*g_Y.T @ (xh-x)

def parameter_gradient_dynamics_2(xh, x, g_ah, g_vh, t):
    pass

# State estimate dynamics
def estimate_dynamics(xh, x, ah, t):
    """
        Dynamics of state estimate x hat
        ph: estimate og p
        qh: estimate of q
        ah: estimate of a
        p: measured p
        q: measured q
    """
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
    """

    g_Y = nabla_Y(xh)

    # Change in state estimate
    if t < t_open:
        return g_Y @ ah + np.hstack((kq*(x[:6]-xh[:6]), kp*(x[6:]-xh[6:])))
    else:
        return g_Y @ ah

def total_dynamics(t, X):
    ah = spnorm_conj_grad(X[24:], norm_order)
    dx = dynamics(t, X[:12])
    dhx = estimate_dynamics(X[12:24], X[:12], ah, t)
    dg_ah = parameter_gradient_dynamics_1(X[12:24], X[:12], t)
    return np.hstack((dx, dhx, dg_ah))

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
if solver == 0:
    sol = solve_ivp(total_dynamics, (0, t_end), X0, max_step=dt)
elif solver == 1:
    sol = forward_euler(total_dynamics, (0, t_end), X0)

# Print solver status
if sol.success:
    status = "succeeded"
else:
    status = "failed"
print(f'IVP solver {status}: {sol.message}')

# Extract states from solution
x = sol.y[:12]
xh = sol.y[12:24]
ah = np.array([spnorm_conj_grad(sol.y[24:, i], norm_order) for i in range(sol.y.shape[1])]).T

print("Final parameter estimate vector:")
ah_final = ah[:,-1]
for i in range(len(ah_final)):
    print(f"{ah_final[i]:.3f}")

"""
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
"""

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
xe = np.sum(np.abs(x - xh), axis=0)
plt.plot(sol.t, xe)
plt.title("Tracking error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{x} - x \Vert_1$")

"""
# Plot parameter estimate error
plt.figure()
ae = np.sum(np.abs(ah.T - a).T, axis=0)
plt.plot(sol.t, ae)
plt.title("Parameter estimation error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{a} - a \Vert_1$")
"""

# Plot parameter penalization function
plt.figure()
breg_div = [bregman_divergence(norm_order, ah_i, ah0) for ah_i in ah.T]
plt.plot(sol.t, breg_div)
plt.title("Bregman Divergence")
plt.xlabel("time, $t$")
plt.ylabel(r"$d_\varphi (\hat{a} \Vert \hat{a}_0)$")

# Plot learning rate
plt.figure()
l_r = [learning_rate(gamma, i, t_d, r_d, theta) for i in range(t_end)]
plt.plot(l_r)
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

# Plot last and first period
period_l = int(6.3259/dt)
plt.figure()
plt.subplot(2,1,1)
for i in range(0,6,2):
    plt.plot(x[i, :period_l], x[i+1, :period_l], color=viridis(i/6))
    plt.plot(xh[i, :period_l], xh[i+1, :period_l], linestyle="--", color=viridis(i/6))
plt.title("First period")
plt.subplot(2,1,2)
for i in range(0,6,2):
    plt.plot(x[i, -period_l:], x[i+1, -period_l:], color=viridis(i/6))
    plt.plot(xh[i, -period_l:], xh[i+1, -period_l:], linestyle="--", color=viridis(i/6))
plt.title("Last period")

plt.show()
