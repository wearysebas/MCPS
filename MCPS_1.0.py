import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import astropy.units as u
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

N = 10
boxW = 10
Nh = int(N/2)
t = 0
tf = 10
dt = 0.1
steps = int(np.ceil((tf)/dt))
plotRealTime = True
acc_x = np.zeros((N,1))
ay = np.zeros((N,1))
az = np.zeros((N,1))
m = 1
rx = np.zeros((N,1))
ry = np.zeros((N,1))
rz = np.zeros((N,1))

def get_initial_coordinates():
  x_coord = np.random.uniform(low=0.0, high=1, size=(N,1))*boxW-0.1
  y_coord = np.random.uniform(low=0.0, high=1, size=(N,1))*boxW-0.1
  z_coord = np.random.uniform(low=0.0, high=1, size=(N,1))*boxW-0.1
  
  return x_coord, y_coord, z_coord
     

def get_initial_velocities():
  x_vel =np.zeros((N,1))#2*(np.random.rand(N,1)-0.5)*boxW
  y_vel =np.zeros((N,1)) #2*(np.random.rand(N,1)-0.5)*boxW
  z_vel =np.zeros((N,1))

  return x_vel, y_vel, z_vel


def get_acc(x,y,z,rx,ry,rz,acc_x,ay,az,rmax,sigma,epsilon):
    #calculate interactions between all particles
    for i in range(N):
        j = 0
        while j < N:
            if i == j:
                j = j+1
            else:
                rx[j]=x[i] - x[j]
                ry[j]=y[i] - y[j]
                rz[j]=z[i] - z[j]
                acc_x[i] = (1/rx[j]) #(4*epsilon*(12*(sigma/r[j]))**11 - 6*(sigma/r[j]**5))/m * np.cos(y[i]/x[i])
                ay[i] = (1/ry[j])
                az[i] = (1/rz[j])
                #acc_x[j] = -rx[i] #(4*epsilon*(12*(sigma/r[j]))**11 - 6*(sigma/r[j]**5))/m * np.cos(y[i]/x[i])
                #ay[j] = -ry[i]
                j = j+1
    return acc_x, ay, az


def move(x,y,z,rx,ry,rz,vx,vy,vz,acc_x,ay,az,t,dt):
    for i in range(N):
        acc_x, ay, az = get_acc(x,y,z,rx,ry,rz,acc_x,ay,az,rmax = 15, sigma = 1, epsilon = 1)
        x[i] += vx[i]*dt + 1/2 * acc_x[i]*(dt**2)
        y[i] += vy[i]*dt + 1/2 * ay[i]*(dt**2)
        z[i] += vz[i]*dt + 1/2 * az[i]*(dt**2)
        vx[i] += acc_x[i]*dt
        vy[i] += ay[i]*dt
        vz[i] += az[i]*dt
        t += dt
    
        if abs(x[i]) >= boxW:
            vx[i] = -vx[i]
            x[i] += vx[i]*dt

        if abs(y[i]) >= boxW:
            vy[i] = -vy[i]
            y[i] += vy[i]*dt

        if abs(z[i]) >= boxW:
            vz[i] = -vz[i]
            z[i] += vz[i]*dt


def simulation(x,y,z,rx,ry,rz,vx,vy,vz,acc_x,ay,az,t,dt):
    for i in range(steps):
        if plotRealTime or (i == steps-1):
            plt.cla()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x,y,z,color='blue', alpha=0.5)
            #plt.scatter(x[Nh:], y[Nh:], z[Nh:],color='red',  alpha=0.5)
            #plt.axis([-boxW,boxW,-boxW,boxW,-boxW,boxW])
            plt.pause(0.001)
        move(x,y,z,rx,ry,rz,vx,vy,vz,acc_x,ay,az,t,dt) 
    plt.xlabel('x')
    plt.ylabel('v')
    plt.show(ax)
                

x, y, z = get_initial_coordinates()
vx, vy, vz = get_initial_velocities()
simulation(x,y,z,rx,ry,rz,vx,vy,vz,acc_x,ay,az,t,dt)