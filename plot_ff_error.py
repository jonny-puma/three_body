import numpy as np
from matplotlib import pyplot as plt


# Load results
path = "data/function_fitting/"
gd_l0_test = np.load(path+"gd_l0_test.npy")
gd_l0_training = np.load(path+"gd_l0_training.npy")
gd_l1_test = np.load(path+"gd_l1_test.npy")
gd_l1_training = np.load(path+"gd_l1_training.npy")
gd_l2_test = np.load(path+"gd_l2_test.npy")
gd_l2_training = np.load(path+"gd_l2_training.npy")
md_l1_test = np.load(path+"md_l1_test.npy")
md_l1_training = np.load(path+"md_l1_training.npy")

plt.figure()

# Plot training error
plt.subplot(1,2,1)
plt.semilogy(gd_l0_training)
plt.semilogy(gd_l1_training)
plt.semilogy(gd_l2_training)
plt.semilogy(md_l1_training)
plt.title("Training error")
plt.xlabel("Iteration")
plt.ylabel(r"$\Vert \hat{y} - y \Vert_2^2$")
plt.ylim(1e-4, 24)

# Plot test error
plt.subplot(1,2,2)
plt.semilogy(gd_l0_test)
plt.semilogy(gd_l1_test)
plt.semilogy(gd_l2_test)
plt.semilogy(md_l1_test)
plt.legend(("GD", "GD $\ell_1$", "GD $\ell_2$", "MD $\ell_1$"))
plt.title("Test error")
plt.xlabel("Iteration")
plt.ylim(1e-4, 24)

plt.subplots_adjust(left=0.05, right=0.95, top=0.91, bottom=0.05)
plt.show()
