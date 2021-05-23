import numpy as np
from matplotlib import pyplot as plt


n_samples = 10
dim_regressor = 200
epochs = 1.2e5
rla = 2e-4
rlb = 2e-4
lp_regulizer = 2
algo = "gd"
norm_order = 1.01

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

# Gradient of squared p-norm
def spnorm_grad(x, p):
    # Define gradient as zero close to x=0 to avoid overflows
    if np.linalg.norm(x, ord=p) < 1e-3:
        return np.zeros(x.shape)
    else:
        return x * (abs(x)/np.linalg.norm(x, ord=p))**(p-2)

# Gradient of squared conjugate p-norm (q-norm)
def spnorm_conj_grad(x, p):
    q = p/(p-1)
    return spnorm_grad(x,q)

np.random.seed(99)
x = np.random.uniform(0, 10, n_samples)
y = f(x)

xt = np.random.uniform(0, 10, n_samples)
yt = f(xt)

a = np.random.uniform(-0.1, 0.1, dim_regressor)
b = np.random.uniform(-10, 0, dim_regressor)
e = np.zeros(epochs)
et = np.zeros(epochs)

if algo == "gd":

    if lp_regulizer == 2:
        lambda_1 = 0.15
        lambda_2 = 0.3
        regulizer = l2_reg
        grad_regulizer = grad_l2
    elif lp_regulizer == 1:
        lambda_1 = 0.005
        lambda_2 = 0.02
        regulizer = l1_reg
        grad_regulizer = grad_l1
    else:
        lambda_1 = 0
        lambda_2 = 0
        regulizer = l1_reg
        grad_regulizer = grad_l1

    # Supertitle for plot
    if lambda_1 == 0 and lambda_2 == 0:
        optimization_details = "Gradient descent"
    else:
        optimization_details = f"""Gradient descent with $\ell_{lp_regulizer}$ regularization
        $\\lambda_1 = {lambda_1}$, $\\lambda_2 = {lambda_2}$"""

    for j in range(epochs):
        d_a = 0
        d_b = 0

        for i in range(n_samples):
            xi = x[i]
            yi = y[i]
            ReLU_i = ReLU(xi + b)
            ReLU_i_a = ReLU_i@a
            G_b_i = grad_Y_b(xi + b)
            d_a += -2*(yi - ReLU_i_a)*ReLU_i
            d_b += -2*(yi - ReLU_i_a)*G_b_i*a
            e[j] += (yi-ReLU_i_a)**2
            et[j] += (yt[i] - ReLU(xt[i] + b)@a)**2

        e[j] /= n_samples
        et[j] /= n_samples
        # e[j] += lambda_1*regulizer(a) + lambda_2*regulizer(b)
        # et[j] += lambda_1*regulizer(a) + lambda_2*regulizer(b)

        a -= rla*(d_a/n_samples + lambda_1*grad_regulizer(a))
        b -= rlb*(d_b/n_samples + lambda_2*grad_regulizer(b))

else:
    
    optimization_details = f"Mirror descent with $\\psi(\\cdot) = \\frac{{1}}{{2}} \\|\\|\\cdot\\|\\|_{{{norm_order:.2f}}}^2$"

    g_a = spnorm_grad(a, norm_order)
    g_b = spnorm_grad(b, norm_order)

    for j in range(epochs):
        a = spnorm_conj_grad(g_a, norm_order)
        b = spnorm_conj_grad(g_b, norm_order)
        d_g_a = 0
        d_g_b = 0

        for i in range(n_samples):
            xi = x[i]
            yi = y[i]
            ReLU_i = ReLU(xi + b)
            ReLU_i_a = ReLU_i@a
            G_b_i = grad_Y_b(xi + b)
            d_g_a += -2*(yi - ReLU_i_a)*ReLU_i
            d_g_b += -2*(yi - ReLU_i_a)*G_b_i*a
            e[j] += (yi-ReLU_i_a)**2
            et[j] += (yt[i] - ReLU(xt[i] + b)@a)**2

        e[j] /= n_samples
        et[j] /= n_samples

        g_a -= rla*d_g_a/n_samples
        g_b -= rlb*d_g_b/n_samples

plt.figure()

plt.subplot(2,2,1)
xp = np.linspace(-2, 12, 100)
yp = np.array([ReLU(xi + b)@a for xi in xp])
plt.plot(xp, yp)
plt.plot(xp, f(xp), linestyle="--")
plt.plot(x, y, "xr")
plt.legend(("Regressor approximation", "True function", "Training samples"))
plt.title("Function approximation")
plt.suptitle(optimization_details)

plt.subplot(2,2,2)
plt.semilogy(e)
plt.semilogy(et)
plt.ylim((1e-4, 30))
plt.legend(("Training error", "Test error"))
plt.title("Squared fitting error")


plt.subplot(2,2,3)
plt.hist(a, bins=50, range=(-0.5, 0.5))
plt.ylim((0, 80))
plt.title(r"$\alpha$ distribution")

plt.subplot(2,2,4)
plt.hist(b, bins=50, range=(-12,2))
plt.ylim((0, 20))
plt.title(r"$\beta$ distribution")

plt.show()

