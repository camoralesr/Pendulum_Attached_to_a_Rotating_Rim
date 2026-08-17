import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# Parámetros del sistema
# -------------------------
g = 9.81
m = 1.0
l = 2.0
R = 1.0
omega = 1

# Condiciones iniciales
phi0 = np.radians(45)
phidot0 = 0

# Integración temporal
tmax = 30
dt = 1e-3
STRIDE = 35
TRAIL = 800
SPEED = 3.0

# -------------------------
# Tamaños de letra
# -------------------------
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14
CLOCK_SIZE = 14

# -----------------------------------------------------
# Dinámica
# phi'' = (R omega^2 cos(phi-omega t) - g sin(phi))/l
# ----------------------------------------------------
def accel(phi, t):
    return (R * omega**2 * cos(phi - omega*t) - g * sin(phi)) / l

def rk4_step(phi, v, t, h):
    k1_phi = v
    k1_v = accel(phi, t)

    k2_phi = v + 0.5*h*k1_v
    k2_v = accel(phi + 0.5*h*k1_phi, t + 0.5*h)

    k3_phi = v + 0.5*h*k2_v
    k3_v = accel(phi + 0.5*h*k2_phi, t + 0.5*h)

    k4_phi = v + h*k3_v
    k4_v = accel(phi + h*k3_phi, t + h)

    phi_new = phi + (h/6)*(k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    v_new = v + (h/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    return phi_new, v_new

# -------------------------
# Integración
# -------------------------
n = int(tmax / dt)
t = np.linspace(0, tmax, n + 1)

phi = np.empty(n + 1)
phidot = np.empty(n + 1)

phi[0] = phi0
phidot[0] = phidot0

for i in range(n):
    phi[i + 1], phidot[i + 1] = rk4_step(phi[i], phidot[i], t[i], dt)

# -------------------------
# Cinemática
# -------------------------
xp = R * cos(omega * t)
yp = R * sin(omega * t)

xb = xp + l * sin(phi)
yb = yp - l * cos(phi)

# -------------------------
# Energía
# -------------------------
vpx = -R * omega * sin(omega * t)
vpy =  R * omega * cos(omega * t)

vrx = l * cos(phi) * phidot
vry = l * sin(phi) * phidot

vbx = vpx + vrx
vby = vpy + vry

K = 0.5 * m * (vbx**2 + vby**2)
U = m * g * yb
E = K + U

# -------------------------
# Figura
# -------------------------
fig, (axp, axe) = plt.subplots(2, 1, figsize=(7, 10),
              gridspec_kw={"height_ratios": [2.2, 1]})

# Panel del péndulo
radio_total = R + l + 0.3
axp.set_xlim(-radio_total, radio_total)
axp.set_ylim(-radio_total, radio_total)
axp.set_aspect("equal")
axp.set_title("Simple Pendulum Attached to a Rotating Rim", fontsize=TITLE_SIZE)
axp.grid(alpha=0.3)
axp.tick_params(axis="both", labelsize=TICK_SIZE)

# Disco guía
theta = np.linspace(0, 2*pi, 400)
axp.plot(R*cos(theta), R*sin(theta), lw=1.2, color="black")
# Radio del disco: línea discontinua
radius_line, = axp.plot([], [], "--", lw=1.2, color="black")
# Barra del péndulo: línea continua
line, = axp.plot([], [], "-", lw=1.2, color="blue")

# Solo el punto correspondiente a la masa del péndulo
bob, = axp.plot([], [], "o", ms=10, color="black")
trace, = axp.plot([], [], "-", lw=1, alpha=0.45, color="red")
clock = axp.text(0.05, 0.93, "", transform=axp.transAxes, fontsize=CLOCK_SIZE)

# Panel de energía
lo = min(K.min(), U.min(), E.min())
hi = max(K.max(), U.max(), E.max())
margen = 0.05 * (hi - lo)
axe.set_xlim(0, tmax)
axe.set_ylim(lo - margen, hi + margen)
axe.set_title("Mechanical Energy as a Function of Time", fontsize=TITLE_SIZE)
axe.set_xlabel("t [s]", fontsize=LABEL_SIZE)
axe.set_ylabel("E [J]", fontsize=LABEL_SIZE)
axe.grid(alpha=0.3)
axe.tick_params(axis="both", labelsize=TICK_SIZE)


lK, = axe.plot([], [], lw=1.1, color="blue", label="K")
lU, = axe.plot([], [], lw=1.1, color="red", label="U")
lE, = axe.plot([], [], lw=1.1, color="green", label="E")
dotE, = axe.plot([], [], "o", ms=3)
axe.legend(loc="upper right", fontsize=LEGEND_SIZE,handlelength=1)

def animate(i):
    radius_line.set_data([0, xp[i]], [0, yp[i]])
    line.set_data([xp[i], xb[i]], [yp[i], yb[i]])
    bob.set_data([xb[i]], [yb[i]])
    #trace.set_data(xb[max(0, i - TRAIL):i], yb[max(0, i - TRAIL):i])
    trace.set_data(xb[:i+1], yb[:i+1])
    clock.set_text(f"t = {t[i]:.2f} s")

    # Energía
    lK.set_data(t[:i+1], K[:i+1])
    lU.set_data(t[:i+1], U[:i+1])
    lE.set_data(t[:i+1], E[:i+1])
    dotE.set_data([t[i]], [E[i]])

    return line, trace, clock, lK, lU, lE, dotE


ani = FuncAnimation(
    fig,
    animate,
    frames=range(0, n + 1, STRIDE),
    interval=STRIDE * dt * 1000 / SPEED,
    blit=True
)

plt.subplots_adjust(hspace=0.35, top=0.93, bottom=0.08)
plt.show()