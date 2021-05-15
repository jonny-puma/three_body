import numpy as np
from matplotlib import pyplot as plt


n_samples = 50
dim_regressor = 500
epochs = 300000
rla = 1e-5
rlb = 1e-5
lp_regulizer = 1
lambda_1 = 0.1
lambda_2 = 0.1

def f(x):
    return x + np.sin(x)

def ReLU(x):
    return (x>0)*x

def grad_Y_b(x):
    return (x>0)*1

def l2_reg(x):
    return np.linalg.norm(x, ord=2)

def l1_reg(x):
    return np.linalg.norm(x, ord=1)

def grad_l2(x):
    return x/np.linalg.norm(x, ord=2)

def grad_l1(x):
    return np.sign(x)

np.random.seed(99)
x = np.random.uniform(0, 10, n_samples)
y = f(x)

a = np.random.uniform(-1, 1, dim_regressor)
b = np.random.uniform(-10, 0, dim_regressor)
e = np.zeros(epochs)

if lp_regulizer == 2:
    regulizer = l2_reg
    grad_regulizer = grad_l2
else:
    regulizer = l1_reg
    grad_regulizer = grad_l1

for j in range(epochs):
    d_a = 0
    d_b = 0

    for i in range(n_samples):
        xi = x[i]
        yi = y[i]
        ReLU_i = ReLU(xi + b)
        G_b_i = grad_Y_b(xi + b)
        d_a += -2*(yi - ReLU_i@a)*ReLU_i
        d_b += -2*(yi - ReLU_i@a)*G_b_i*a
        e[j] += (yi-ReLU_i@a)**2/n_samples

    e[j] += lambda_1*regulizer(a) + lambda_2*regulizer(b)
    a -= rla*(d_a/n_samples + lambda_1*grad_regulizer(a))
    b -= rlb*(d_b/n_samples + lambda_2*grad_regulizer(b))

plt.figure()
plt.plot(e)

plt.figure()
plt.hist(a, bins=30)
plt.title(r"$\alpha$ distribution")

plt.figure()
plt.hist(b, bins=30)
plt.title(r"$\beta$ distribution")

plt.figure()
xp = np.linspace(0, 10, 100)
yp = np.array([ReLU(xi + b)@a for xi in xp])
plt.plot(xp, yp)
plt.plot(xp, f(xp))
plt.plot(x, y, "xr")
plt.show()

