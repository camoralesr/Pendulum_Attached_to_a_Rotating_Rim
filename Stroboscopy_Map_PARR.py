import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters: PARR
# -----------------------------------------------------
g = 9.81
m = 1
l = 2
R = 1
L = R/l
kappa = 0.01
om = np.sqrt(g/(l*kappa))
T = 2*pi

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Trans = 0
Nperiods = 500
ST = 300
dt = T/ST

# -----------------------------------------------------
# Initial conditions: grid
# -----------------------------------------------------
phi0 = np.deg2rad(np.arange(-180, 181, 20))
v0 = np.arange(-5, 6, 1)/om
PHI0, V0 = np.meshgrid(phi0, v0, indexing="ij")
phi01 = PHI0.ravel()
v01 = V0.ravel()
n = len(phi01)
y = np.concatenate([phi01, v01])

# -----------------------------------------------------
# Dynamics: dimensionless PARR
# -----------------------------------------------------
def dyn(t, y):
    phi = y[:n]
    v = y[n:]
    dphi = v
    dv = -kappa*sin(phi) + L*cos(phi-t)
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
# Angular wrapping: [alpha-pi, alpha+pi)
# -----------------------------------------------------
alpha = pi/3
def wrap(theta):
    return (theta - alpha + pi) % (2*pi) - pi + alpha

# -----------------------------------------------------
# Stroboscopic storage
# -----------------------------------------------------
ns = Nperiods - Trans + 1
phis = np.empty((ns, n))
vs = np.empty((ns, n))
s = 0
if Trans == 0:
    phis[s] = wrap(y[:n])
    vs[s] = y[n:]
    s += 1

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
Ts = Nperiods * ST
for step in range(Ts):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    if (step + 1) % ST == 0:
        completed_period = (step + 1)//ST
        if completed_period >= max(1, Trans):
            phis[s] = wrap(y[:n])
            vs[s] = y[n:]
            s += 1

# ---------------------------------------------------------
# Colors according to initial condition & Stroboscopic map
# ---------------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
oc = [colors[i % len(colors)] for i in range(n)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n):
    ax.scatter(phis[:s, i], vs[:s, i], s=1, color=oc[i],
        linewidths=0, rasterized=True)
ax.set_xlim(alpha-pi, alpha+pi)
#ax.set_xlabel(r"$\phi$", fontsize=24)
#ax.set_ylabel(r"$\dot\phi$", fontsize=24)
ax.tick_params(axis="both", labelsize=16)
ax.set_box_aspect(1)
ax.text(0.03, 0.97, rf"$\kappa = {kappa:.2f}$",
    transform=ax.transAxes, ha="left", va="top", fontsize=16)
ax.text(0.97, 0.97, rf"$(d)$",
    transform=ax.transAxes, ha="right", va="top", fontsize=16)
plt.tight_layout()
plt.savefig("sk001.pdf", format="pdf", bbox_inches="tight", dpi=800)
plt.show()

# -----------------------------------------------------
# Numerical results
# -----------------------------------------------------
print("\n" + "="*55)
print("       PARR NUMERICAL SIMULATION RESULTS")
print("="*55)
print(f"Gravity                  = {g:.4f} m/s^2")
print(f"Pendulum length          = {l:.4f} m")
print(f"Rotating rim radius      = {R:.4f} m")
print(f"Dimensionless L          = {L:.4f}")
print(f"Kappa                    = {kappa:.6f}")
print(f"Angular frequency        = {om:.6f} rad/s")
print(f"Dimensionless period     = {T:.6f}")
print(f"Integration step         = {dt:.6f}")
print(f"Steps per period         = {ST}")
print(f"Total RK4 steps          = {Ts}")
print(f"Transient periods        = {Trans}")
print(f"Number of periods        = {Nperiods}")
print(f"Initial conditions       = {n}")
print(f"Points per condition     = {s}")
print(f"Total stored points      = {s*n}")
print(f"Final dimensionless time = {Nperiods*T:.4f}")
print(f"Final physical time      = {Nperiods*T/om:.4f} s")
print("="*55)
