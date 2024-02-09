import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def norm(x):
	return x/la.norm(x)

def reject(v, n):
    return v - np.dot(v,n)*n

# Setting initial conditions
dt = 0.01  # time step
tmax = 25  # simulation time
glim = 20  # maximum acceleration

mass = 175  # missile mass

T = np.array([0, 0, 10000])  # target position
M = np.array([-10000, 0, 10000])  # missile position

Vt = np.array([0, 300, 0])  # target velocity
Vm = np.array([343*5, 0, 0])  # missile velocity

def At(t):  # target acceleration
    return np.array([0, 0, np.cos(t)*30])

def Am(T, M, Vt, Vm, t):
    N = 3
    R = T - M  # Relative position vector from missile to target
    Vr = Vt - Vm  # Relative velocity vector between target and missile
    
    # Avoid division by zero or very small numbers in LOS rate calculation
    if np.dot(R, R) < 1e-6:
        return np.array([0, 0, 0])  # Prevent potential overflow
    
    # Compute LOS rate as the rate of change of the line of sight angle
    LOS_rate = np.cross(R, Vr) / la.norm(R)**2
    
    #accel = -N * la.norm(Vr) * np.cross(norm(Vm),LOS_rate) #PN
    
    tgo = la.norm(R) / la.norm(Vr)
    zem = R+Vr*tgo
    zemi=reject(zem,norm(R))
    
    accel = N*zemi/tgo**2
    
    accel = accel + N*reject(At(t),norm(R))/2 #APN
    
    accel = reject(accel, norm(Vm)) #control surface
    
    #bound the acceleration to avoid overflow
    
    if la.norm(accel) > glim*9.81:
        return glim * norm(accel)
    
    return accel-(la.norm(accel)**2*norm(Vm)*0.5)/mass

def quadraticDrag(V, rho=1.2, Cd=0.5, A=0.125):
    return (-0.5 * rho * Cd * A * la.norm(V) * V)/mass

def getrho(h,p0=101325, T0=288.15, L=0.00976, R=8.31447, M=0.0289644, g=9.81, Rs=287):
    return (p0*(1-L*h/T0)**(g*M/(R*L)))/(Rs*T0)

timelist = []


Tlist = []
Mlist = []

Vmlist = []

TGOlist = []
ZEMlist = []

t = 0
while t < tmax:
    timelist.append(t)
    
    Tlist.append(T)
    Mlist.append(M)
    
    Vmlist.append(Vm)
    
    TGOlist.append(la.norm(T-M)/la.norm(Vt-Vm))
    ZEMlist.append((Vt-Vm)*(la.norm(T-M)/la.norm(Vt-Vm)))
    
    # Update missile and target positions
    M = M + Vm * dt
    T = T + Vt * dt

    # Update missile and target velocities
    Vm = Vm + Am(T, M, Vt, Vm, t) * dt + quadraticDrag(Vm, rho=getrho(M[2])) * dt
    Vt = Vt + At(t) * dt
    
    if np.dot(Vm,T-M)<0:
        break

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

ax2.plot(timelist, TGOlist, label='TGO-est', color='blue')
#actual tgo (line with -45 deg slope until the missile is closest to the target)
ax2.plot(np.arange(0, min_range*dt, dt), np.arange(min_range*dt, 0, -dt), label='TGO-actual', color='cyan')

ax2.set_ylabel('TGO (seconds)')
ax2.set_xlabel('Time (seconds)')

ax3 = ax2.twinx()
ax3.plot(timelist, [la.norm(ZEM) for ZEM in ZEMlist], label='ZEM', color='red')
ax3.set_ylabel('ZEM (meters)')

#velocity plot
ax2.plot(timelist, [la.norm(Vm)/343 for Vm in Vmlist], label='Vm', color='yellow')

#add  vertical line to show the time when the missile is closest to the target
ax2.axvline(min_range*dt, color='green', linestyle='--', label='t final')
#add horizontal line to show the minimum range
ax3.axhline(min_range_sep, color='yellow', linestyle='--', label='min range')
#add label to the line showing the minimum range
ax3.text(1, min_range_sep+200, f'min range: {min_range_sep:.1f} (m)', fontsize=12, color='yellow')

fig2.legend()

plt.show()



#make animation of the 3d engagement
import matplotlib.animation as animation

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

Mline, = ax.plot([],[],[], label='Missile')
Tline, = ax.plot([],[],[], label='Target')
ax.set(xlim=(-5000, 1000), ylim=(0, 5000), zlim=(9500, 10500))

def init():
    Mline.set_data([], [])
    Mline.set_3d_properties([])
    Tline.set_data([], [])
    Tline.set_3d_properties([])
    return Mline, Tline

def animate(i,Mline,Tline):
    Mline.set_data(Mlist[:i,0], Mlist[:i,1])
    Mline.set_3d_properties(Mlist[:i,2])
    Tline.set_data(Tlist[:i,0], Tlist[:i,1])
    Tline.set_3d_properties(Tlist[:i,2])
    return Mline, Tline

anim=animation.FuncAnimation(fig, animate, init_func=init, frames=len(Tlist), fargs=(Mline,Tline), interval=0.25, blit=True)

plt.show()
