import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm(x):
	return x/la.norm(x)

# Setting initial conditions
dt = 0.01  # time step
tmax = 30  # simulation time
glim = 3  # maximum acceleration

T = np.array([0, 0, 0])  # target position
M = np.array([-10000, 0, 0])  # missile position

Vt = np.array([0, 300, 0])  # target velocity
Vm = np.array([600, 0, 0])  # missile velocity

def At(t):  # target acceleration
    return np.array([0, 0, np.cos(t/3)*30])

def Am(T, M, Vt, Vm, t):
    # Proportional navigation guidance law, simplified to avoid overflow
    N = -3
    R = T - M  # Relative position vector from missile to target
    Vr = Vt - Vm  # Relative velocity vector between target and missile
    
    # Avoid division by zero or very small numbers in LOS rate calculation
    if np.dot(R, R) < 1e-6:
        return np.array([0, 0, 0])  # Prevent potential overflow
    
    # Compute LOS rate as the rate of change of the line of sight angle
    LOS_rate = np.cross(R, Vr) / la.norm(R)**2
    
    # Calculate acceleration using proportional navigation guidance law
    # Assuming the missile acceleration is proportional to the LOS rate and the magnitude of the relative velocity
    accel = N * la.norm(Vr) * np.cross(norm(Vm),LOS_rate)
    
    #bound the acceleration to avoid overflow
    
    accel = np.clip(accel, -glim*9.8, glim*9.8)
    
    return accel


Tlist = []
Mlist = []

TGOlist = []
ZEMlist = []

t = 0
while t < tmax:
    Tlist.append(T)
    Mlist.append(M)
    
    TGOlist.append(la.norm(T-M)/la.norm(Vt-Vm))
    ZEMlist.append((Vt-Vm)*(la.norm(T-M)/la.norm(Vt-Vm)))
    
    # Update missile and target positions
    M = M + Vm * dt
    T = T + Vt * dt

    # Update missile and target velocities
    Vm = Vm + Am(T, M, Vt, Vm, t) * dt
    Vt = Vt + At(t) * dt

    # Update time
    t += dt

# Plotting 3D

#dark mode
plt.style.use('dark_background')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Tlist = np.array(Tlist)
Mlist = np.array(Mlist)
#find index min range of T-M
min_range = np.argmin(la.norm(Tlist-Mlist, axis=1))
min_range_sep = la.norm(Tlist[min_range]-Mlist[min_range])

ax.plot(Tlist[:, 0], Tlist[:, 1], Tlist[:, 2], label='Target')
ax.plot(Mlist[:, 0], Mlist[:, 1], Mlist[:, 2], label='Missile')
#insert a point to show the time when the missile is closest to the target
ax.scatter(Tlist[min_range, 0], Tlist[min_range, 1], Tlist[min_range, 2], color='green', label='t final')
ax.scatter(Mlist[min_range, 0], Mlist[min_range, 1], Mlist[min_range, 2], color='green')

ax.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot() #dual axis plot TGO and ZEM

ax2.plot(np.arange(0, tmax, dt), TGOlist, label='TGO-est', color='blue')
#actual tgo (line with -45 deg slope until the missile is closest to the target)
ax2.plot(np.arange(0, min_range*dt, dt), np.arange(min_range*dt, 0, -dt), label='TGO-actual', color='cyan')

ax2.set_ylabel('TGO (seconds)')
ax2.set_xlabel('Time (seconds)')

ax3 = ax2.twinx()
ax3.plot(np.arange(0, tmax, dt), [la.norm(ZEM) for ZEM in ZEMlist], label='ZEM', color='red')
ax3.set_ylabel('ZEM (meters)')

#add  vertical line to show the time when the missile is closest to the target
ax2.axvline(min_range*dt, color='green', linestyle='--', label='t final')
#add horizontal line to show the minimum range
ax3.axhline(min_range_sep, color='yellow', linestyle='--', label='min range')
#add label to the line showing the minimum range
ax3.text(1, min_range_sep+200, f'min range: {min_range_sep:.1f}', fontsize=12, color='yellow')

fig2.legend()

plt.show()




#make animation of the 3d engagement
import matplotlib.animation as ani

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Tlist = np.array(Tlist)
Mlist = np.array(Mlist)

def update(i):
    ax.plot(Tlist[i, 0], Tlist[i, 1], Tlist[i, 2], label='Target')
    ax.plot(Mlist[i, 0], Mlist[i, 1], Mlist[i, 2], label='Missile')
    ax.legend()
    
ani = ani.FuncAnimation(fig, update, frames=len(Tlist), repeat=True)

plt.show()