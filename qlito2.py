import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import astropy.units as u
#import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import math

#NOTES
#All rows in each matrix represent one of the coordinates (x,y,z)
#All columns in each maatrix represent each particle's coordinates 
## pa nosotros los pendejos: Pos[0][:] = renglón 0 de la matriz


#plasma density (part/vol)
Rho = 0.001  
#box width
boxW = 10  
#num de dimensiones
dim = 3 
#Box Volume
Vol = (2*boxW)**3 
# number of particles 
N = int(Rho * Vol) 
#Nh = int(N/2)
#Start time 
t = 0 
#End Time
tf = 10 
# time differential
dt = 0.1 
# simulation steps
steps = int(np.ceil((tf)/dt)) 
plotRealTime = True
# Accelerations in each axis
acc_x = np.zeros((1,N)) 
ay = np.zeros((1,N)) 
az = np.zeros((1,N)) 
# Particle mass
m = 1 
#Distance in each axis
rx = np.zeros((1,N)) 
ry = np.zeros((1,N)) 
rz = np.zeros((1,N))

#atomic radius
radius = 0.5

#Position matrix
Pos = np.zeros((dim,N))

#Distance matrix
Dis = np.zeros((dim,N))

#Velocity matrix
Vel = np.zeros((dim,N))

#Acceleration matrix
Acc = np.zeros((dim,N))

def get_initial_coordinates():
  x_coord = np.random.uniform(low=0.0, high=1, size=(1,N))*boxW-0.1
  y_coord = np.random.uniform(low=0.0, high=1, size=(1,N))*boxW-0.1
  z_coord = np.random.uniform(low=0.0, high=1, size=(1,N))*boxW-0.1

  Pos[0] = x_coord
  Pos[1] = y_coord
  Pos[2] = z_coord
  
  return Pos
     
def check_overlap(Pos):
    
    i = 0
    j = 1

    while(i<N-1):
        while(j<N):
            if(i == j):
                j = j + 1
            dist = math.sqrt((Pos[0,i] -Pos[0,j])**2 + (Pos[1,i] -Pos[1,j])**2 + (Pos[2,i] -Pos[2,j])**2)
            if(dist>(2*radius)):
                j = j + 1
            else:
                new_x = np.random.uniform(low=0.0, high=1, size=(1))*boxW-0.1
                new_y = np.random.uniform(low=0.0, high=1, size=(1))*boxW-0.1
                new_z = np.random.uniform(low=0.0, high=1, size=(1))*boxW-0.1

                print("*****TRASLAPE*******")
                Pos[0,i] = new_x
                Pos[1,i] = new_y
                Pos[2,i] = new_z
                
                j = 0
        i = i + 1
        j = i + 1


    return None



def get_initial_velocities():
  x_vel =np.zeros((1,N))#2*(np.random.rand(1,N)-0.5)*boxW
  y_vel =np.zeros((1,N)) #2*(np.random.rand(1,N)-0.5)*boxW
  z_vel =np.zeros((1,N))

  Vel[0] = x_vel
  Vel[1] = y_vel
  Vel[2] = z_vel

  return Vel


def get_acc(Pos,Dis,Acc): #FUERA DE SERVICIO HASTA NUEVO AVISO
    #calculate interactions between all particles
    for i in range(N):
        j = 0
        while j < N: 
            if i == j:
                j = j+1 
            else:
                rx[0,j]=Pos[0,i] - Pos[0,j]
                ry[0,j]=Pos[1,i] - Pos[1,j]
                rz[0,j]=Pos[2,i] - Pos[2,j]

                #La aceleración de la part. i esta dada por la distancia a la part. j
                Acc[0,i] = (1/rx[0,j]) 
                Acc[1,i] = (1/ry[0,j])
                Acc[2,i] = (1/rz[0,j])
                j = j+1
    return Acc


def move(Pos,Dis,Vel,Acc,t,dt):
    for i in range(N):
        Acc[0,:], Acc[1,:], Acc[2,:] = get_acc(Pos,Dis,Acc)
        Pos[0,i] += Vel[0,i]*dt + 1/2 * Acc[0,i]*(dt**2)
        Pos[1,i] += Vel[1,i]*dt + 1/2 * Acc[1,i]*(dt**2)
        Pos[2,i] += Vel[2,i]*dt + 1/2 * Acc[2,i]*(dt**2)
        Vel[0,i] += Acc[0,i]*dt
        Vel[1,i] += Acc[1,i]*dt
        Vel[2,i] += Acc[2,i]*dt
        t += dt
    
        if abs(Pos[0,i]) >= boxW:
            Vel[0,i] = -Vel[0,i]
            x[i] += Vel[0][i]*dt

        if abs(Pos[1][i]) >= boxW:
            Vel[1,i] = -Vel[1,i]
            Pos[1,i] += Vel[1,i]*dt

        if abs(Pos[2][i]) >= boxW:
            Vel[2,i] = -Vel[2,i]
            Pos[2,i] += Vel[2,i]*dt


def simulation(Pos,Dis,Vel,Acc,t,dt):
    for i in range(steps):
        if plotRealTime or (i == steps-1):
            #plt.cla()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(Pos[0,:],Pos[1,:],Pos[2,:],s = radius*10, color='blue', alpha=0.5)
            #plt.axis([-boxW,boxW,-boxW,boxW,-boxW,boxW])
            plt.pause(0.001)
        move(Pos,Dis,Vel,Acc,t,dt) 
        plt.close()
    plt.xlabel('x')
    plt.ylabel('v')
    plt.show()
                

x, y, z = get_initial_coordinates()
check_overlap(Pos)
vx, vy, vz = get_initial_velocities()
simulation(Pos,Dis,Vel,Acc,t,dt)
