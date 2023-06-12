import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import astropy.units as u
import matplotlib.animation as animation
import math
import plasmapy

#particle - parcile - particle - mesh
#NOTES
#All rows in each matrix represent one of the coordinates (x,y,z)
#All columns in each maatrix represent each particle's coordinates 
## pa nosotros los pendejos: M[0][:] = renglón 0 de la matriz  
# Columna 0 de la matriz M[:,0]


# comentarios
##Partes que se quieren probar
###Posibles formas de optimizar


#Define particles
p = plasmapy.particles.Particle('p+')
e = plasmapy.particles.Particle('e-')

#Coulomb Constant
k = 100#14.39

#plasma density (part/vol)
Rho = 0.001  
#box width
boxW = 10
#num de dimensiones
dim = 3 
#Box Volume
Vol = (2*boxW)**3 
# number of particles 
N = 2#int(Rho * Vol/1000)
#Nh = int(N/2)
#Start time 
t = 0 
#End Time
tf = 10 
# time differential
dt = 0.1 
# simulation steps
steps = int(np.ceil((tf)/dt)) 
#plotRealTime = True
# Accelerations in each axis
acc_x = np.zeros((1,N)) 
ay = np.zeros((1,N)) 
az = np.zeros((1,N)) 
# Particle mass
mass = 1 
#Distance in each axis
rx = np.zeros((1,N)) 
ry = np.zeros((1,N)) 
rz = np.zeros((1,N))

n = range(N)

#atomic radius
radius = 0.5

# Assign the desired value for the coupling constant g
g = 1

# Assign the desired value for the constant alpha
alpha = 1

#Position matrix
Pos = np.zeros((dim,N))

#Distance matrix
Dis = np.zeros((dim,N))

#Velocity matrix
Vel = np.zeros((dim,N))

#Acceleration matrix
Acc = np.zeros((dim,N))

#Forces matrix
F_y = np.zeros((dim,N))
F_c = np.zeros((dim,N))
F_l = np.zeros((dim,N))
F_og = np.zeros((dim,N))
Force = np.zeros((N,N),dtype=object)

#Force matrix: ij es la fuerza en las tres componentes de la partícula i debído a la partícula j.
Force_tot = np.zeros((dim,N),dtype=object)

#Plasma matrix, first three rows are the position of each particle, next row is the charge of particle, final row is mass.
params = 1 #numbers of parameters in the plasma
Plas = np.zeros((dim + params,N))

#Color scheme
color = []

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
            dist = math.sqrt((Pos[0,i] - Pos[0,j])**2 + (Pos[1,i] - Pos[1,j])**2 + (Pos[2,i] - Pos[2,j])**2)
            if(dist>(2*radius)):
                j = j + 1
            else:
                new_x = np.random.uniform(low=0.0, high=1, size=(1))*boxW-0.1
                new_y = np.random.uniform(low=0.0, high=1, size=(1))*boxW-0.1
                new_z = np.random.uniform(low=0.0, high=1, size=(1))*boxW-0.1

                #print("*****TRASLAPE*******")
                Pos[0,i] = new_x
                Pos[1,i] = new_y
                Pos[2,i] = new_z
                
                j = 0
        i = i + 1
        j = i + 1


    return None

def get_distance(Pos):  #Crea una matriz donde las componentes i,j es la distancia entre las partículas i-j. Cada componente son  3 coordenadas, representando la distancia de las particulas en cada coordenada. 
    # M[renglon,columna][coordenada] para acceder a la información.
    Dis = np.zeros((N,N),dtype=object)
    for i in range(N):
            for j in range(N):
                Dis[i,j]= Pos[:,j] - Pos[:,i]
                ##Dis[i,j]=Dis[i,j][:,np.newaxis] #creates column vector
                ##print(Dis)
                ### np.linalg.norm(Dis[i,j]) checar si es más rápido hacer la matriz con normas de distancia o calcularlas en cada instancia.
    return Dis

def get_initial_velocities():
  x_vel =np.zeros((1,N))#2*(np.random.rand(1,N)-0.5)*boxW
  y_vel =np.zeros((1,N)) #2*(np.random.rand(1,N)-0.5)*boxW
  z_vel =np.zeros((1,N))

  Vel[0] = x_vel
  Vel[1] = y_vel
  Vel[2] = z_vel

  return Vel

#Gives particles in the plasma their characteristics
def create_plasma():
    Plas[0,:] = Pos[0,:]
    Plas[1,:] = Pos[1,:]
    Plas[2,:] = Pos[2,:]
    for i in range(N): #para que el plasma sea neutro p = e ----> 3er renglón es la carga de la partícula
        if i % 2 == 1:
            Plas[3,i] = e.charge_number
            color.append("blue")
        else:
            Plas[3,i] = p.charge_number
            color.append("red")

    return None

def Forces(Pos, Vel, Yuk, Coulomb, Lorentz, Ohm_Gen):
    
    Dis = get_distance(Pos)

    #if Yuk == True:
    #    F_y = np.zeros((dim,N))

    if Coulomb == True:
        for i in range(N):
            j = 0
            while j < N: #Not consider auto interactions
                if i == j:
                    Force[i,j] = np.zeros((1,3))
                    j = j+1 
                else: 
                    if np.linalg.norm(Dis[i,j]) >= 2*radius ** np.linalg.norm(Dis[i,j]) <= boxW/2:
                        #La aceleración de la part. i esta dada por la distancia a la part. j. La última parte es el vector director.
                        Force[i,j] = -k*Plas[3,i] * Plas[3,j] / np.dot(Dis[i,j],Dis[i,j].transpose()) * Dis[i,j]/np.linalg.norm(Dis[i,j]) 
                        j = j+1
                    else:
                        Force[i,j] = np.zeros((1,3))
                        #Vel[i] = np.zeros((1,3))
                        j = j+1

            F_c[:,i] = np.sum(Force[i,:])


    #if Lorentz == True:
        F_l = np.zeros((dim,N))

    #if Ohm_Gen == True:      
    #    F_og = np.zeros((dim,N))   

    Force_tot = F_c + F_og + F_l + F_y
    Acc = Force_tot / mass #of particle i   

    return Acc

def move(Pos,Vel,Acc,t,dt):
    #print("positions", Pos)
    Acc = Forces(Pos,Vel, Yuk=False, Coulomb=True, Lorentz=False, Ohm_Gen=True)
    Vel = Vel + Acc*dt
    Pos = Pos + Vel*dt + 1/2 * Acc*(dt**2)
    t += dt
    #print("Acc", Acc)
    #print("Vel", Vel)
    #print("Final Pos", Pos)
    for i in range(N):     #corregir
    
        if Pos[0,i] >= boxW or Pos[0,i] <= -boxW :
            Vel[0,i] = -Vel[0,i]
            Pos[0,i] += Vel[0,i]*dt

        if Pos[1,i] >= boxW or Pos[1,i] <= -boxW:
            Vel[1,i] = -Vel[1,i]
            Pos[1,i] += Vel[1,i]*dt

        if Pos[2,i] >= boxW or Pos[2,i] <= -boxW:
            Vel[2,i] = -Vel[2,i]
            Pos[2,i] += Vel[2,i]*dt
    #print(Pos)
    return Pos

def animate(n):
    global Pos
    Pos = move(Pos,Vel,Acc,t,dt)
    graph._offsets3d = (Pos[0,:], Pos[1,:], Pos[2,:])


get_initial_coordinates()
check_overlap(Pos)
vx, vy, vz = get_initial_velocities()
print(color)
create_plasma()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-boxW, boxW)
ax.set_ylim3d(-boxW, boxW)
ax.set_zlim3d(-boxW, boxW)
graph = ax.scatter3D(Pos[0,:], Pos[1,:], Pos[2,:], c = color[:], s = 32*radius)
ani = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False)

plt.show()
