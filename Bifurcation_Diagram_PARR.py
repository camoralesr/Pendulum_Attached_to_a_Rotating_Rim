import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
delta = 0.1
l = 0.5
T = 2*pi
phi0 = 1
v0 = 2

# -----------------------------------------------------
# Kappa values
# -----------------------------------------------------
kmin = 0
kmax = 1.5
dk= 0.001
ks = np.arange(kmin, kmax + dk/2, dk)
n = len(ks)

# -----------------------------------------------------
# Numerical method parameters
# -----------------------------------------------------
Trans = 500
Nkeep = 500
steps_per_T = 300
dt = T/steps_per_T

# -----------------------------------------------------
# Initial state
# -----------------------------------------------------
phi = np.full(n, phi0)
v = np.full(n, v0)
y = np.concatenate([phi, v])

# -----------------------------------------------------
# Dynamics: damped PARR
# -----------------------------------------------------
def dyn(t, y):
    phi = y[:n]
    v = y[n:]
    dphi = v
    dv = (-delta*v - ks*sin(phi) + l*cos(phi-t))
    return np.concatenate([dphi, dv])

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
# Angular wrapping: [0, 2*pi)
# -----------------------------------------------------
def wrap(phi):
    return phi % (2*pi)

# -----------------------------------------------------
# Storage for stroboscopic points
# -----------------------------------------------------
phi_strobe = np.empty((Nkeep, n))
v_strobe = np.empty((Nkeep, n))
save_index = 0

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
total_periods = Trans + Nkeep
total_steps = total_periods*steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    if (step+1) % steps_per_T == 0:
        completed_period = (step+1)//steps_per_T
        if completed_period > Trans:
            phi_strobe[save_index] = wrap(y[:n])
            v_strobe[save_index] = y[n:]
            save_index += 1

# -----------------------------------------------------
# Bifurcation diagram & Figure format
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(np.tile(ks, Nkeep), phi_strobe.ravel(), s=0.2,
    color="blue", linewidths=0, rasterized=True)
ax.set_xlabel(r"$\kappa$", fontsize=12)
ax.set_ylabel(r"$\phi$", fontsize=12)
ax.tick_params(axis="both", labelsize=10)
ax.text(0.02, 0.98, rf"$\delta = {delta}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=10)
ax.text(0.98, 0.98, rf"$\ell = {l}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
ax.set_xlim(kmin, kmax)
ax.set_ylim(0, 2*pi)
ax.set_box_aspect(0.6)
plt.tight_layout()

# -----------------------------------------------------
# Save figure
# -----------------------------------------------------
plt.savefig("BifPARR.pdf", format="pdf", bbox_inches="tight", dpi=800)
plt.show()

# -----------------------------------------------------
# Numerical results
# -----------------------------------------------------
print("\n" + "="*55)
print("       DAMPED PARR BIFURCATION DIAGRAM")
print("="*55)
print(f"Damping coefficient       = {delta}")
print(f"Dimensionless length      = {l}")
print(f"Kappa range               = [{kmin}, {kmax}]")
print(f"Kappa increment           = {dk}")
print(f"Stroboscopic period       = {T:.6f}")
print(f"Integration step          = {dt:.6f}")
print(f"Steps per period          = {steps_per_T}")
print(f"Transient periods         = {Trans}")
print(f"Stored periods            = {Nkeep}")
print(f"Initial angle             = {phi0}")
print(f"Initial angular velocity  = {v0}")
print(f"Number of kappa values    = {n}")
print(f"Total stored points       = {save_index*n}")
print(f"Total RK4 steps           = {total_steps}")
print("="*55)
