import numpy as np
from numpy import sin, cos, pi, log
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters: PARR
# -----------------------------------------------------
delta = 0.1
l = 0.5
T = 2*pi
phi0 = 1.0
v0 = 2.0
t = 0

# -----------------------------------------------------
# Kappa values
# -----------------------------------------------------
kmin = 0
kmax = 1.5
dk = 0.001
ks = np.arange(kmin, kmax + dk/2, dk)
n = len(ks)

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
nTrans = 500
nLyap = 500
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Initial state & unit tangent vector
# -----------------------------------------------------
phi = np.full(n, phi0)
v = np.full(n, v0)
xi = np.full(n, 1.0/np.sqrt(2.0))
eta = np.full(n, 1.0/np.sqrt(2.0))
y = np.concatenate([phi, v, xi, eta])

# -----------------------------------------------------
# Dynamics + variational equations
# -----------------------------------------------------
def dyn(t, y):
    phi = y[:n]
    v = y[n:2*n]
    xi = y[2*n:3*n]
    eta = y[3*n:]
    dphi = v
    dv = (-delta*v - ks*sin(phi) + l*cos(phi - t))
    dxi = eta
    deta = (-(ks*cos(phi) + l*sin(phi - t))*xi - delta*eta)
    return np.concatenate([dphi, dv, dxi, deta])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# -----------------------------------------------------
# Integration: transient regime
# -----------------------------------------------------
for period in range(nTrans):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
        xi = y[2*n:3*n]
        eta = y[3*n:]
        tangent_norm = np.hypot(xi, eta)
        y[2*n:3*n] = xi/tangent_norm
        y[3*n:] = eta/tangent_norm

# -----------------------------------------------------
# Maximum Lyapunov exponent
# -----------------------------------------------------
sum_log = np.zeros(n)
for period in range(nLyap):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
        xi = y[2*n:3*n]
        eta = y[3*n:]
        tangent_norm = np.hypot(xi, eta)
        sum_log += log(tangent_norm)
        y[2*n:3*n] = xi/tangent_norm
        y[3*n:] = eta/tangent_norm

# -----------------------------------------------------
# Maximum Lyapunov exponent per unit time
# -----------------------------------------------------
lambda_max = sum_log/(nLyap*T)
imax = np.argmax(lambda_max)
kappa_at_max = ks[imax]
lambda_at_max = lambda_max[imax]
print(f"Maximum point = " f"({kappa_at_max:.6f}, {lambda_at_max:.6f})")

# -----------------------------------------------------
# MLE & Figure format
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ks, lambda_max, linewidth=0.5, color='blue')
ax.axhline(y=0, linewidth=0.2, color='black', linestyle='--')
ax.set_xlabel(r'$\kappa$', fontsize=12)
ax.set_ylabel(r'$\lambda_{\max}$', fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.text(0.02, 0.98, rf"$\delta = {delta}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10)
ax.text(0.98, 0.9, rf'$\kappa_m = {kappa_at_max:.2f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=10)
ax.text(0.98, 0.98, rf"$\ell = {l}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
ax.set_xlim(ks[0], ks[-1])
ax.set_box_aspect(0.65)
plt.tight_layout()

# -----------------------------------------------------
# Save figure
# -----------------------------------------------------
plt.savefig('MLE.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
