import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import astropy.units as u
import matplotlib.animation as animation
import math
import plasmapy
from scipy.interpolate import griddata
from plasmapy.plasma import grids

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
p_rad = 0.85 * 10**(-15)
e = plasmapy.particles.Particle('e-')
e_rad = 10**(-22)
D = plasmapy.particles.Particle('D+')
D_rad = 2.1421 * 10**(-15)
T = plasmapy.particles.Particle('T+')
T_rad = 1.7591 * 10**(-15)
neutron = plasmapy.particles.Particle('n')
n_rad = 0.8 * 10**(-15)
alpha_particle=plasmapy.particles.Particle('α')
alpha_particle_rad = 1.6782 * 10**(-15)

#Coulomb Constant
k = 100#14.39

#Para crear el toro
#Radio mayor del toro
R = 50
#Radio menor del toro
r = 25

#plasma density (part/vol)
Rho = 0.001  
#box width
boxW = 50
#num de dimensiones
dim = 3 
#Box Volume
Vol = (2*boxW)**3 
# number of particles 
N = 50#int(Rho * Vol/1000)
#Nh = int(N/2)
#Start time 
t = 0 
#End Time
tf = 10 
# time differential
dt = 0.01 
# simulation steps
steps = int(np.ceil((tf)/dt)) 
#plotRealTime = True
# Accelerations in each axis
acc_x = np.zeros((1,N)) 
ay = np.zeros((1,N)) 
az = np.zeros((1,N)) 
#Particle mass
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
alpha_constant = 1

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

#Create initial EM fields
E = np.zeros((dim,N))
B = np.zeros((dim,N))

#Initialize grid
# Grid size
x_start, x_end, num_points_x = -boxW, boxW, 5
y_start, y_end, num_points_y = -boxW, boxW, 5
z_start, z_end, num_points_z = -boxW, boxW, 5

# Create the grid points
x = np.linspace(x_start * u.cm, x_end * u.cm, num_points_x)
y = np.linspace(y_start * u.cm, y_end * u.cm, num_points_y)
z = np.linspace(z_start * u.cm, z_end * u.cm, num_points_z)

# Create the grid using meshgrid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid = grids.CartesianGrid(X, Y, Z)
#Confirm grid is created correctly
print(f"Is the grid uniformly spaced? {grid.is_uniform}") 

#Give initial E and B fields
Ex = np.random.rand(*grid.shape) * u.V / u.m
Ey = np.random.rand(*grid.shape) * u.V / u.m
Ez = np.random.rand(*grid.shape) * u.V / u.m
Bx = np.random.rand(*grid.shape) * u.T
By = np.random.rand(*grid.shape) * u.T
Bz = np.random.rand(*grid.shape) * u.T

#Add quantitites to the grid (create vector field)
grid.add_quantities(B_x=Bx, B_y=By, B_z=Bz)
grid.add_quantities(E_x=Ex, E_y=Ey, E_z=Ez)


#Plasma matrix, first three rows are the position of each particle, next row is the charge of particle, final row is mass.
params = 2 #numbers of parameters in the plasma
Plas = np.zeros((dim + params,N))

#Color scheme
color = []

geometry = "torus"

def get_initial_coordinates(geometry, R, r):

    if(geometry == "torus"):
        x_coord = np.random.uniform(low=0.0, high=1, size=(1,N))
        r = r*x_coord
        theta = np.random.uniform(low=0.0, high=1, size=(1,N))
        phi = np.random.uniform(low=0.0, high=1, size=(1,N))

        theta = theta*2*np.pi
        phi = phi*2*np.pi

        Pos[0] = np.sin(theta)*(R + r*np.cos(phi))
        Pos[1] = np.cos(theta)*(R + r*np.cos(phi))
        Pos[2] = r*np.sin(phi)

    elif(geometry == "cartesian"):

        x_coord = np.random.uniform(low=0.0, high=1, size=(1,N))*boxW-0.1
        y_coord = np.random.uniform(low=0.0, high=1, size=(1,N))*boxW-0.1
        z_coord = np.random.uniform(low=0.0, high=1, size=(1,N))*boxW-0.1
    
        Pos[0] = x_coord
        Pos[1] = y_coord
        Pos[2] = z_coord

    return Pos
     
def check_overlap(Pos, geometry, R, r):
    
    i = 0
    j = 1
    if (geometry == "cartesian"):
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
    elif (geometry == "torus"):
        while(i<N-1):
            while(j<N):
                if(i == j):
                    j = j + 1
                dist = math.sqrt((Pos[0,i] - Pos[0,j])**2 + (Pos[1,i] - Pos[1,j])**2 + (Pos[2,i] - Pos[2,j])**2)
                if(dist>(2*radius)):
                    j = j + 1
                else:
                    x_coord = np.random.uniform(low=0.0, high=1, size=(1))
                    r = r*x_coord
                    theta = np.random.uniform(low=0.0, high=1, size=(1))
                    phi = np.random.uniform(low=0.0, high=1, size=(1))

                    theta = theta*2*np.pi
                    phi = phi*2*np.pi

                    Pos[0,i] = np.sin(theta)*(R + r*np.cos(phi))
                    Pos[1,i] = np.cos(theta)*(R + r*np.cos(phi))
                    Pos[2,i] = r*np.sin(phi)
                    
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
  x_vel =np.random.uniform(low=0.0, high=1, size=(1,N))*boxW#np.random.uniform(low=0.0, high=1, size=(1,N))
  y_vel =np.random.uniform(low=0.0, high=1, size=(1,N))*boxW#np.random.uniform(low=0.0, high=1, size=(1,N))
  z_vel =np.random.uniform(low=0.0, high=1, size=(1,N))*boxW#np.random.uniform(low=0.0, high=1, size=(1,N))

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
        #falta revisión
        if i % 2 == 0:
            Plas[3,i] = e.charge_number
            Plas[4,i] = e.mass.value
            color.append("blue")
        elif i % 3 == 0:
            Plas[3,i] = D.charge_number
            Plas[4,i] = D.mass.value
            color.append("green")
        elif i % 5 == 0:
            Plas[3,i] = T.charge_number
            Plas[4,i] = T.mass.value
            color.append("pink")
        else:
            Plas[3,i] = p.charge_number
            Plas[4,i] = p.mass.value
            color.append("red")
           
    return Plas


def solve_em_fields():

    return Nothing

def interpolate_EM_fields(Pos):
    Ex_vals, Ey_vals, Ez_vals = grid.volume_averaged_interpolator(Pos.transpose(), "E_x", "E_y", "E_z", persistent=True)

    E[0] = Ex_vals
    E[1] = Ey_vals
    E[2] = Ez_vals

    Bx_vals, By_vals, Bz_vals = grid.volume_averaged_interpolator(Pos.transpose(), "B_x", "B_y", "B_z", persistent=True)

    B[0] = Bx_vals
    B[1] = By_vals
    B[2] = Bz_vals
    
    #This function returns the values of the E and B fields in the positions of each particle.
    return E, B

def Forces(Dis, Vel, Yuk, Coulomb, Lorentz, Ohm_Gen):
    
    #E, B = interpolate_EM_fields(Pos)


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


    if Lorentz == True:
        for i in range(N):
            F_l[:,i] = Plas[3,i]*E[:,i] + Plas[3,i]*np.cross(Vel[:,i], B[:,i])

    #if Ohm_Gen == True:      
    #    F_og = np.zeros((dim,N))   

    Force_tot = F_c + F_og + F_l + F_y
    Acc = Force_tot / mass #of particle i   

    return Acc

def move(Pos,Vel,Acc,t,dt,Plas):
    Dis = get_distance(Pos)
    Acc = Forces(Dis,Vel, Yuk=False, Coulomb=False, Lorentz=True, Ohm_Gen=True)
    Vel = Vel + Acc*dt

    #fusion reaction
    Plas[0,:] = Pos[0,:]
    Plas[1,:] = Pos[1,:]
    Plas[2,:] = Pos[2,:] 

    #boundary conditions    
    for i in range(0,N):
        if ((np.sqrt(Pos[0,i]**2 + Pos[1,i]**2) - R)**2 + Pos[2,i]**2  >= (r-radius)**2):
           Acc[0,i] = 0
           Acc[1,i] = 0
           Acc[2,i] = 0
           Vel[0,i] = 0#-Vel[0,i]
           Vel[1,i] = 0#-Vel[1,i]
           Vel[2,i] = 0#-Vel[2,i]

    Pos = Pos + Vel*dt + 1/2 * Acc*(dt**2)

    for i in range(N):                                                                                                 
        for j in range(N): 
            if (Plas[4,i]==D.mass.value and Plas[4,j]==T.mass.value) and  Dis[i,j].any() < T_rad + D_rad:
                E_k = 1/2 * Plas[4,i] * (Vel[0,i]**2 + Vel[1,i]**2 + Vel[2,i]**2) * 6.241*10**(19)
                print(E_k)
                Plas[3,i]=alpha_particle.charge_number 
                Plas[4,i]=alpha_particle.mass.value
                color[i]=("orange")
                Plas[3,j]=neutron.charge_number 
                Plas[4,j]=n.mass.value
                color[j]=("grey")
               # Here should come something about velocities #

    t += dt


    return Pos, Vel, Plas

def create_ovito(Plas):
    global Pos
    global Vel 
    global N
    global t 

    outputF = open("outputiF.xyz","w")
    atom='C'
    for paso in range(steps):
        outputF.write(str(N)+'\n')
        outputF.write('Lattice=" 10.00 0 0 0  10.00 0 0 0 10.00"')
        outputF.write('\n')
        for every  in range(N):
            x, y, z = map(float, (Pos[0,every], Pos[1,every], Pos[2,every]))
            if Plas[3,every] == e.charge_number:
                 atom='C'
            elif Plas[3,every] == p.charge_number:
                 atom='O'
            elif Plas[3,every] == alpha_particle.charge_number:
                 atom='N'
            else:
                 atom='Fe'

            outputF.write("{}\t {}\t {}\t {}\t".format(atom,x,y,z))
            outputF.write("\n")
        t += dt
        Pos, Vel, Plas = move(Pos,Vel,Acc,t,dt,Plas)

    outputF.close()

def show_vect_field():
    Bequis = grid.__getitem__("B_x")
    Bye = grid.__getitem__("B_y")
    Bzeta = grid.__getitem__("B_z")

    Eequis = grid.__getitem__("E_x")
    Eye = grid.__getitem__("E_y")
    Ezeta = grid.__getitem__("E_z")

    ax.scatter(X, Y, Z, color='green', alpha=0.2)
    ax.quiver(X, Y, Z, Eequis, Eye, Ezeta, length=5, color='pink')
    ax.quiver(X, Y, Z, Bequis, Bye, Bzeta, length=5, color='cyan')
    


def animate(n):
    global Pos
    global Vel
    global Plas
    Pos, Vel , Plas = move(Pos,Vel,Acc,t,dt,Plas)
    graph._offsets3d = (Pos[0,:], Pos[1,:], Pos[2,:])
    create_ovito(Plas)


resolution = 100  # Number of points to sample

# Create the parameter values for the torus
theta = np.linspace(0, 2 * np.pi, resolution)
phi = np.linspace(0, 2 * np.pi, resolution)
theta, phi = np.meshgrid(theta, phi)

# Compute the coordinates of the points on the torus
equis = (R + r * np.cos(phi)) * np.cos(theta)
yee = (R + r * np.cos(phi)) * np.sin(theta)
zeta = r * np.sin(phi)




get_initial_coordinates(geometry, R, r)
check_overlap(Pos, geometry, R, r)
Plas=create_plasma()
vx, vy, vz = get_initial_velocities()
print(color)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Plot the torus surface
ax.plot_surface(equis, yee, zeta, cmap='viridis', alpha = 0.1)
ax.set_xlim3d(-boxW, boxW)
ax.set_ylim3d(-boxW, boxW)
ax.set_zlim3d(-boxW, boxW)
graph = ax.scatter3D(Pos[0,:], Pos[1,:], Pos[2,:], c = color[:], s = 32*radius)
#Following command shows grid and the next the vectorfield for E and B
#show_vect_field()
ani = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False)


plt.show()