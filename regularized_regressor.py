import pdb
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
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
    if np.linalg.norm(x, ord=p) < 1e-12:
        return np.zeros(x.shape)
    else:
        return x * (abs(x)/np.linalg.norm(x, ord=p))**(p-2)

# Gradient of squared conjugate p-norm (q-norm)
def spnorm_conj_grad(x, p):
    q = p/(p-1)
    return spnorm_grad(x,q)

# Bregman divergence with p-norm as potential function
def bregman_divergence(norm_order, x1, x2):
    return l2(x1, ord=norm_order) - l2(x2, ord=norm_order) - spnorm_grad(x2, norm_order) @ (x1-x2)

# Time dependent learning rate
def learning_rate(t):
    if t < t_d:
        return gamma*((1-w_t) + w_t*np.cos(theta*t))
    elif t < t_end_learning:
        w_cos = w_t/(1+r_d*(t-t_d))
        return gamma*(1-w_cos + w_cos*np.cos(theta*t))/(1+r_d*(t-t_d))
    else:
        return 0

# Simulation parameters
dt = 0.01
t_end = 150

# Integration algorithm: 0 = scipy, 1 = forward Euler
solver = 0

# First or second order adaption law
adaption_law = 0

# Parameter estimator parameters
gamma = 3.5
mu = 3.5
beta = 3.5
kp = 5
kq = 5
t_d = t_end
t_end_learning = t_end
t_open = t_end
r_d = 0.1
theta = 0.3
w_t = 0
norm_order = 1.05

# Physical constants
g = 1
m1, m2, m3 = 1, 1, 1

# Initial conditions for figure 8 periodic solution
x1, x2 =  (-0.97000436, 0.24308753), (0, 0)
v1, v2 = (0.4662036850, 0.4323657300), (-0.93240737, -0.86473146)
x3, v3 = (-x1[0], -x1[1]), v1
period = 6.3259

# Initial conditions for butterfly periodic solution
v1 = (0.30689, 0.12551)
# Isosceles colinear configuration
x1 = (-1, 0)
x2 = (1, 0)
x3 = (0, 0)
v2 = v1
v3 = (-2*v1[0], -2*v1[1])
period = 6.2356

# Pack particle states into system state
x0 = np.array((x1, x2, x3, v1, v2, v3)).flatten()

# True parameters
a = np.array((0, 0, 0, 0, 0, 0, 1/(2*m1), 1/(2*m2), 1/(2*m3),
                    0, 0, 0, g*m1*m2, g*m2*m3, g*m1*m3, 0, 0, 0, 0, 0, 0))

# Initial conditions for estimates
xh0 = x0
"""
if norm_order >= 1 and norm_order <= 2:
    ah0 = np.zeros(21)
else:
    ah0 = 0.1*np.ones(21)
"""
ah0 = 0.5*np.ones(21)
g_ah0 = spnorm_grad(ah0, norm_order)
g_vh0 = g_ah0

# Initial condition for total simulation state
if adaption_law == 0:
    X0 = np.hstack((x0, xh0, g_ah0))
else:
    X0 = np.hstack((x0, xh0, g_ah0, g_vh0))

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
    Y_ = np.zeros(21)
    for i in range(3):
        q_i = x[i*2:i*2+2]
        p_i = x[i*2+6:i*2+8]
        Y_[i] = q_i@q_i
        Y_[i+3] = (q_i@q_i)**2
        Y_[i+6] = p_i@p_i
        Y_[i+9] = (p_i@p_i)**2
        
        j = (i+1)%3
        q_j = x[j*2:j*2+2]
        q_ij = 1.0/l2(q_i - q_j)
        Y_[i+12] = -q_ij
        Y_[i+15] = -q_ij**2
        Y_[i+18] = -q_ij**3
    return Y_
    
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

    del_Y[6:,12:] = np.array([[-gf12_1[0], -gf13_1[0],          0, -gf12_2[0], -gf13_2[0],          0, -gf12_3[0], -gf13_3[0],          0],
                              [-gf12_1[1], -gf13_1[1],          0, -gf12_2[1], -gf13_2[1],          0, -gf12_3[1], -gf13_3[1],          0],
                              [ gf12_1[0],          0, -gf23_1[0],  gf12_2[0],          0, -gf23_2[0],  gf12_3[0],          0, -gf23_3[0]],
                              [ gf12_1[1],          0, -gf23_1[1],  gf12_2[1],          0, -gf23_2[1],  gf12_3[1],          0, -gf23_3[1]],
                              [         0,   gf13_1[0], gf23_1[0],          0,  gf13_2[0],  gf23_2[0],          0,  gf13_3[0],  gf23_3[0]],
                              [         0,   gf13_1[1], gf23_1[1],          0,  gf13_2[1],  gf23_2[1],          0,  gf13_3[1],  gf23_3[1]]])
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

# First order adaption law
def parameter_gradient_dynamics_1(x, xh, t):
    """
        Dynamics of parameter estimate gradient
        ph: p hat, estimate of p
        qh: q hat, estimagte of q
        pe: estimate error for p
        qe: estimate error for q
    """
    # Regressor basis jacobian
    g_Y = nabla_Y(x)

    # Time dependent learning rate
    r_l = learning_rate(t)
    return -r_l*g_Y.T @ (xh-x)

# Second order adaption law
def parameter_gradient_dynamics_2(x, xh, g_ah, g_vh, t):
    r_l = learning_rate(t)
    g_Y = nabla_Y(x)
    N = 1 + mu*l2(Yh)**2
    dg_vh = -r_l*g_Y @ (xh - x)
    dg_ah = beta*N*(g_vh - g_ah)
    return dg_ah, dg_vh

# State estimate dynamics
def estimate_dynamics(xh, x, ah, t):
    # Regressor basis jacobian
    g_Y = nabla_Y(x)

    # Open control loop after t_open
    if t < t_open:
        return g_Y @ ah + np.hstack((kq*(x[:6]-xh[:6]), kp*(x[6:]-xh[6:])))
    else:
        return g_Y @ ah

# Simulator state dynamics
def total_dynamics(t, X):
    ah = spnorm_conj_grad(X[24:45], norm_order)
    dx = dynamics(t, X[:12])
    dhx = estimate_dynamics(X[12:24], X[:12], ah, t)
    if adaption_law == 1:
        dg_ah, dg_vh = parameter_gradient_dynamics_2(X[:12], X[12:24], X[24:45], X[45:], t)
        return np.hstack((dx, dhx, dg_ah, dg_vh))
    else:
        dg_ah = parameter_gradient_dynamics_1(X[:12], X[12:24], t)
        return np.hstack((dx, dhx, dg_ah))

# Gravitational force between particles
def gforce_1(x1, x2, m):
    return g*m*(x1-x2)/(l2(x1-x2)**3)

def gforce_2(x1, x2, m):
    return 2*g*m*(x1-x2)/(l2(x1-x2)**4)

def gforce_3(x1, x2, m):
    return 3*g*m*(x1-x2)/(l2(x1-x2)**5)

# Three body system Hamiltonian
def hamiltonian(x):
    return (-g*m1*m2/l2(x[0:2]-x[2:4]) - g*m2*m3/l2(x[2:4]-x[4:6]) - g*m1*m3/l2(x[0:2]-x[4:6])
            + l2(x[6:8])**2/(2*m1) + l2(x[8:10])**2/(2*m2) + l2(x[10:12])**2/(2*m3))

# Solve IVP
if solver == 0:
    sol = solve_ivp(total_dynamics, (0, t_end), X0, max_step=dt, method="Radau")
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
ah = np.array([spnorm_conj_grad(sol.y[24:45, i], norm_order) for i in range(sol.y.shape[1])]).T

# Set all a < 1e-3 to 0
# ah[:,-1] *= ah[:,-1] > 1e-3

# Print parameter vector at t_end
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
plt.ylim(-0.5, 3)
plt.xlabel("time, $t$")
plt.ylabel("$\hat{a}$")

# Plot histogram of final parameter values
plt.figure()
plt.hist(ah[:,-1], bins=20, range=(-0.2, 1.2))
plt.ylim(0, 16)
plt.title("Parameter distribution")
plt.xlabel("Parameter value")
plt.ylabel("Frequency")

# Plot tracking error
plt.figure()
# xe = np.sum(np.abs(x - xh), axis=0)
xe = l2(x-xh, axis=0)
plt.semilogy(sol.t, xe)
plt.title("Tracking error")
plt.ylim(1e-4, 0.7)
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{x} - x \Vert_2$")

"""
# Plot parameter estimate error
plt.figure()
ae = np.sum(np.abs(ah.T - a).T, axis=0)
plt.plot(sol.t, ae)
plt.title("Parameter estimation error")
plt.xlabel("time, $t$")
plt.ylabel(r"$\Vert \hat{a} - a \Vert_1$")

# Plot parameter bregman divergence from initial condition
plt.figure()
breg_div = [bregman_divergence(norm_order, ah_i, ah0) for ah_i in ah.T]
plt.plot(sol.t, breg_div)
plt.title("Bregman Divergence")
plt.xlabel("time, $t$")
plt.ylabel(r"$d_\varphi (\hat{a} \Vert \hat{a}_0)$")

# Plot learning rate
plt.figure()
l_r = [learning_rate(i) for i in range(t_end)]
plt.plot(l_r)
plt.title("Learning rate")
plt.xlabel("$t$")
plt.ylabel(r"$\gamma$")
"""

# Plot last and first period
period_l = int(period/dt)
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

plt.figure()
plt.subplot(2,1,1)
hamil = np.array([hamiltonian(x[:,i]) for i in range(len(sol.t))])
hamil_h = np.array([Y(xh[:,i])@ah[:,-1] for i in range(len(sol.t))])
plt.plot(sol.t, hamil)
plt.plot(sol.t, hamil_h)
plt.legend((r"$\mathcal{H}$", r"$\hat{\mathcal{H}}$"))
plt.title("Hamiltonian trajectory")

# Plot Hamiltonian error
plt.subplot(2,1,2)
# he = hamiltonian(xh) - hamiltonian(sol.y)
plt.plot(sol.t, hamil - hamil_h) 
plt.title("Hamiltonian error")
plt.xlabel("$t$")
plt.ylabel(r"$\hat{\mathcal{H}} - \mathcal{H}$")

# Surfplot of Hamiltonian potential with respect to position for q1
fig = plt.figure()
ax = fig.add_subplot(2,2,1, projection="3d")
axlim = 2.5
resolution = 0.01
q1 = np.arange(-axlim+0.5, axlim+0.5, resolution)
q2 = np.arange(-axlim, axlim, resolution)
X_plot, Y_plot = np.meshgrid(q1, q2)
Z1_plot = np.zeros(X_plot.shape)
zlim1 = (-5, 0)

for i in range(X_plot.shape[0]):
    for j in range(X_plot.shape[1]):
        Z1_plot[j,i] = hamiltonian(np.hstack((q1[i], q2[j], x[2:,-1])))

Z1_plot = np.clip(Z1_plot, zlim1[0], zlim1[1])
cnorm1 = matplotlib.colors.Normalize(vmin=zlim1[0], vmax=zlim1[1])
ax.plot_surface(X_plot, Y_plot, Z1_plot,
                norm=cnorm1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(zlim1[0], zlim1[1])
plt.title("Hamiltonian potential")
plt.xlabel("$q_{1,1}$")
plt.ylabel("$q_{1,2}$")

# Surfplot of estimated Hamiltonian potential with respect to position for particle 1
ax = fig.add_subplot(2, 2, 2, projection="3d")
Z2_plot = np.zeros(X_plot.shape)

for i in range(X_plot.shape[0]):
    for j in range(X_plot.shape[1]):
        Z2_plot[j,i] = Y(np.hstack((q1[i], q2[j], x[2:,-1]))) @ ah[:,-1]

if norm_order == 2:
    Z2_max = np.max(Z2_plot)
    zlim2 = (Z2_max - 5, Z2_max)
else:
    zlim2 = zlim1
Z2_plot = np.clip(Z2_plot, zlim2[0], zlim2[1])
cnorm2 = matplotlib.colors.Normalize(vmin=zlim2[0], vmax=zlim2[1])
ax.plot_surface(X_plot, Y_plot, Z2_plot,
                norm=cnorm2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(zlim2[0], zlim2[1])
plt.title("Estimated Hamiltonian potential")
plt.xlabel("$q_{1,1}$")
plt.ylabel("$q_{1,2}$")

# Surfplot of Hamiltonian with respect to momentum for particle 1
ax = fig.add_subplot(2, 2, 3, projection="3d")
p1 = np.arange(-axlim, axlim, resolution)
p2 = np.arange(-axlim, axlim, resolution)
X_plot, Y_plot = np.meshgrid(p1, p2)
Z3_plot = np.zeros(X_plot.shape)

for i in range(X_plot.shape[0]):
    for j in range(X_plot.shape[1]):
        Z3_plot[j,i] = hamiltonian(np.hstack((x0[:6], p1[i], p2[j], x[8:,-1])))

Z3_min = np.min(Z3_plot)
zlim3 = (Z3_min, Z3_min + 5)
Z3_plot = np.clip(Z3_plot, zlim3[0], zlim3[1])
cnorm3 = matplotlib.colors.Normalize(vmin=zlim3[0], vmax=zlim3[1])
ax.plot_surface(X_plot, Y_plot, Z3_plot,
                norm=cnorm3, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(zlim3[0], zlim3[1])
plt.title("Hamiltonian momentum")
plt.xlabel("$p_{1,1}$")
plt.ylabel("$p_{1,2}$")

# Surfplot of estimated Hamiltonian with respect to momentum for particle 1
ax = fig.add_subplot(2, 2, 4, projection="3d")
Z4_plot = np.zeros(X_plot.shape)

for i in range(X_plot.shape[0]):
    for j in range(X_plot.shape[1]):
        Z4_plot[j,i] = Y(np.hstack((x0[:6], p1[i], p2[j], x[8:,-1]))) @ ah[:,-1]


Z4_min = np.min(Z4_plot)
zlim4 = (Z4_min, Z4_min + 5)
Z4_plot = np.clip(Z4_plot, zlim4[0], zlim4[1])
cnorm4 = matplotlib.colors.Normalize(vmin=zlim4[0], vmax=zlim4[1])
ax.plot_surface(X_plot, Y_plot, Z4_plot,
                norm=cnorm4, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
ax.set_zlim(zlim4[0], zlim4[1])
plt.title("Estimated Hamiltonian momentum")
plt.xlabel("$p_{1,1}$")
plt.ylabel("$p_{1,2}$")

# Plot 3D Hamiltonian potential error
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
bias = np.average(Z4_plot - Z3_plot)
zlim5 = (-2.5, 2.5)
cnorm5 = matplotlib.colors.Normalize(vmin=-1.8, vmax=1.8)
Z5_plot = Z2_plot - Z1_plot - bias
ax.plot_surface(X_plot, Y_plot, Z5_plot,
        norm=cnorm5, cmap=cm.seismic, linewidth=0, antialiased=False, zorder=0)
ax.set_zlim(zlim5[0], zlim5[1])
plt.title(f"Hamiltonian potential error $\\ell_{{{norm_order}}}$")
plt.xlabel("$q_{1,1}$")
plt.ylabel("$q_{1,2}$")

plt.show()
