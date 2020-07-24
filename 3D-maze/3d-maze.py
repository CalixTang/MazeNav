import vpython as vp
import random
import random
from math import *
import time
from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats


import threading
import time

#3D version of the 2D Maze class
class Maze:
    def __init__(self, height, width, length):
        self.height = height
        self.width = width
        self.length = length
        self.genGrid(height, width, length)
        
    def genGrid(self,height, width, length):
        self.grid = [[[Cell(pos = [i,j,h], walls = [True,True,True,True,True,True]) for h in range(length)] for j in range(width)] for i in range(height)]
        self.grid[0][0][0].walls[0] = False 
        self.grid[-1][-1][-1].walls[5] = False 
        
    def genMaze(grid, start, end):
        #print(start)
        if Maze.checkGrid(grid) == True:
            return
        else:
            grid[end[0]][end[1]][end[2]].inMaze = True #add the end to the maze
            begin = start[:] #we'll use this during cutting
            #Random walk starting at the start (this should take a while)
            while not grid[start[0]][start[1]][start[2]].inMaze:
                grid[start[0]][start[1]][start[2]].dir = Maze.randomWalk([start[0],start[1],start[2]], len(grid), len(grid[0]), len(grid[0][0]))
                if grid[start[0]][start[1]][start[2]].dir == 0:
                    start[0] -= 1
                    pass
                elif grid[start[0]][start[1]][start[2]].dir == 1:
                    start[1] -= 1
                    pass
                elif grid[start[0]][start[1]][start[2]].dir == 2:
                    start[2] -= 1
                    pass
                elif grid[start[0]][start[1]][start[2]].dir == 3:
                    start[1] += 1
                    pass
                elif grid[start[0]][start[1]][start[2]].dir == 4:
                    start[2] += 1
                    pass
                elif grid[start[0]][start[1]][start[2]].dir == 5:
                    start[0] += 1
                    pass
            #begin "cutting" a path out starting from the beginning following dir, put each cell that it follows into the maze
            while not begin[0] == end[0] and begin[1] == end[1] and begin[2] == end[2]:
                grid[begin[0]][begin[1]][begin[2]].walls[grid[begin[0]][begin[1]][begin[2]].dir] = False
                grid[begin[0]][begin[1]][begin[2]].inMaze = True
                if grid[begin[0]][begin[1]][begin[2]].dir == 0:
                    begin[0] -= 1
                    grid[begin[0]][begin[1]][begin[2]].walls[5] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 1:
                    begin[1] -= 1
                    grid[begin[0]][begin[1]][begin[2]].walls[3] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 2:
                    begin[2] -= 1
                    grid[begin[0]][begin[1]][begin[2]].walls[4] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 3:
                    begin[1] += 1
                    grid[begin[0]][begin[1]][begin[2]].walls[1] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 4:
                    begin[2] += 1
                    grid[begin[0]][begin[1]][begin[2]].walls[2] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 5:
                    begin[0] += 1
                    grid[begin[0]][begin[1]][begin[2]].walls[0] = False
            #call genRandom now because you have a first path! :D
            Maze.genMazeR(grid)

    def genMazeR(grid):
        if Maze.checkGrid(grid) == True:
            return
        else:
            #find a random unMazed cell to start
            x = random.randint(0,len(grid)-1)
            y = random.randint(0,len(grid[0])-1)
            z = random.randint(0,len(grid[0][0])-1)
            while grid[x][y][z].inMaze:
                x = random.randint(0,len(grid)-1)
                y = random.randint(0,len(grid[0])-1)
                z = random.randint(0,len(grid[0][0])-1)
            begin = [x,y,z]
            #random walk until we hit a cell that's in the maze (yes, x and y are "flipped" here)
            while not grid[x][y][z].inMaze:
                grid[x][y][z].dir = Maze.randomWalk([x,y,z], len(grid), len(grid[0]),len(grid[0][0]))
                if grid[x][y][z].dir == 0:
                    x -= 1
                elif grid[x][y][z].dir == 1:
                    y -= 1
                elif grid[x][y][z].dir == 2:
                    z -= 1
                elif grid[x][y][z].dir == 3:
                    y += 1    
                elif grid[x][y][z].dir == 4:
                    z += 1
                elif grid[x][y][z].dir == 5:
                    x += 1
            #cut
            while not grid[begin[0]][begin[1]][begin[2]].inMaze:
                grid[begin[0]][begin[1]][begin[2]].walls[grid[begin[0]][begin[1]][begin[2]].dir] = False
                grid[begin[0]][begin[1]][begin[2]].inMaze = True
                if grid[begin[0]][begin[1]][begin[2]].dir == 0:
                    begin[0] -= 1
                    grid[begin[0]][begin[1]][begin[2]].walls[5] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 1:
                    begin[1] -= 1
                    grid[begin[0]][begin[1]][begin[2]].walls[3] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 2:
                    begin[2] -= 1
                    grid[begin[0]][begin[1]][begin[2]].walls[4] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 3:
                    begin[1] += 1
                    grid[begin[0]][begin[1]][begin[2]].walls[1] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 4:
                    begin[2] += 1
                    grid[begin[0]][begin[1]][begin[2]].walls[2] = False
                elif grid[begin[0]][begin[1]][begin[2]].dir == 5:
                    begin[0] += 1
                    grid[begin[0]][begin[1]][begin[2]].walls[0] = False
            #recur
            Maze.genMazeR(grid)
    
    #outputs a random direction to go in for a cell input in terms of its position in a 2D array. 
    def randomWalk(pos, height, width, length):
        t0 = random.randint(0,5)
        #print(str(pos) + ' ' + str(height) + ' ' + str(width) + ' ' + str(t0))
        if (t0 == 0 and pos[0] == 0): 
            return Maze.randomWalk(pos, height, width, length)
        if (t0 == 1 and pos[1] == 0):
            return Maze.randomWalk(pos, height, width, length)            
        if (t0 == 2 and pos[2] == 0): 
            return Maze.randomWalk(pos, height, width, length)
        if (t0 == 3 and pos[1] == width-1):
            return Maze.randomWalk(pos, height, width, length)
        if (t0 == 4 and pos[2] == length-1):
            return Maze.randomWalk(pos, height, width, length)
        if (t0 == 5 and pos[0] == height-1):
            return Maze.randomWalk(pos, height, width, length)
        else:
            return t0
    
    def checkGrid(grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                for k in range(len(grid[0][0])):
                    if not grid[i][j][k].inMaze:
                        return False
        return True
    
#A cell is a 1x1x1 volume with six walls around it by default, marked by ULFRBD (up, left, front, right, back, down) directions. It has a defined position vector.
class Cell:
    
    
    def __init__(self, pos, walls):
        self.pos = pos
        self.walls = walls
        #self.visited = False
        self.dir = -1 # 0, 1, 2, 3, 4, or 5 when in maze
        self.inMaze = False
   

maze = Maze(height = 5, width = 6, length = 7)
Maze.genMaze(grid = maze.grid, start = [0,0,0], end = [4,5,6])
maze.grid[0][0][0].walls[0] = False
maze.grid[4][5][6].walls[5] = False

#Main - Drawing each cell by the walls.
scene = vp.canvas(width = 1200, height = 1000, title = '3D Maze')
vp.sphere(pos = vp.vector(0,0,0),radius = 0.1, color = vp.vector(1,1,1)) #Origin
vp.sphere(pos = vp.vector(0,5,7),radius = 0.2, color = vp.vector(1,0,0)) #start
vp.sphere(pos = vp.vector(6,0,0),radius = 0.2, color = vp.vector(0,0,1)) #end


quadArray = [] #walls will be added here
for i in range(len(maze.grid)):
    for j in range(len(maze.grid[0])):
        for k in range(len(maze.grid[0][0])):
            #Using if statements for each wall, draw the ones that exist.
            abspos = vp.vector(maze.grid[i][j][k].pos[1], maze.height - maze.grid[i][j][k].pos[0], maze.length - maze.grid[i][j][k].pos[2])
            cellcolor = vp.vector(random.random(),random.random(),random.random())
            DBL = vp.vertex(pos = abspos + vp.vector(0,-1,-1), color = cellcolor)
            DBR = vp.vertex(pos = abspos + vp.vector(1,-1,-1), color = cellcolor)
            DFR = vp.vertex(pos = abspos + vp.vector(1,-1,0), color = cellcolor)
            DFL = vp.vertex(pos = abspos + vp.vector(0,-1,0), color = cellcolor)
            UFL = vp.vertex(pos = abspos, color = cellcolor)
            UFR = vp.vertex(pos = abspos + vp.vector(1,0,0), color = cellcolor)
            UBR = vp.vertex(pos = abspos + vp.vector(1,0,-1), color = cellcolor)
            UBL = vp.vertex(pos = abspos + vp.vector(0,0,-1), color = cellcolor)
            if maze.grid[i][j][k].walls[0]:
                quadArray.append(vp.quad(vs = [UFL,UFR,UBR,UBL]))
            if maze.grid[i][j][k].walls[1]:
                quadArray.append(vp.quad(vs = [UFL,UBL,DBL,DFL]))
            if maze.grid[i][j][k].walls[2]:
                quadArray.append(vp.quad(vs = [UFL,UFR,DFR,DFL]))
            if maze.grid[i][j][k].walls[3]:
                quadArray.append(vp.quad(vs = [UFR,DFR,DBR,UBR]))
            if maze.grid[i][j][k].walls[4]:
                quadArray.append(vp.quad(vs = [UBL,UBR,DBR,DBL]))
            if maze.grid[i][j][k].walls[5]:
                quadArray.append(vp.quad(vs = [DFL,DFR,DBR,DBL]))

def create_uniform_particles(x_range, y_range, z_range, xB_range, yB_range, zB_range, N):
    particles = np.empty((N, 6))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(z_range[0], z_range[1], size=N)
    particles[:, 3] = uniform(xB_range[0], xB_range[1], size = N)
    particles[:, 4] = uniform(yB_range[0], yB_range[1], size = N)
    particles[:, 5] = uniform(zB_range[0], zB_range[1], size = N)
    for i in range(N):
        bearing = particles[i, 3:6]
        particles[i, 3:6] = bearing / np.linalg.norm(bearing)
    return particles
    
def predict(particles, u, std, dt=1.):
    """ move according to control input u [dx/dt,dy/dt,dz/dt,dtheta/dt,dphi/dt]
    with noise Q """
    
    N = len(particles)
    # move in the (noisy) commanded direction
    dist = [(u[0] * dt) + (randn(N) * std[0]),(u[1] * dt) + (randn(N) * std[1]),(u[2] * dt) + (randn(N) * std[2]),(u[3] * dt) + (randn(N) * std[3]),(u[4] * dt) + (randn(N) * std[4]),(u[5] * dt) + (randn(N) * std[5]) ]
    particles[:, 0] += dist[0]
    particles[:, 1] += dist[1]
    particles[:, 2] += dist[2]
    particles[:, 3] += dist[3]
    particles[:, 4] += dist[4]
    particles[:, 5] += dist[5]
    for i in range(len(particles)):
        bearing = particles[i, 3:6]
        particles[i, 3:6] /= np.linalg.norm(bearing)
    
#Update - update weights based on measurement - no idea what I'm doing here.
def update(particles, weights, z, R):
    euc_distance = np.linalg.norm(particles[:,0:3] - z[0:3] + randn(3)*R[0:3], axis=1)
    bearing_distance = np.linalg.norm(particles[:,3:6] - z[3:6] + randn(3)*R[3:6], axis=1)
    weights *= 1/(1+euc_distance+bearing_distance + 1.e-300)
    weights /= sum(weights) # normalize
    
def neff(weights):
    return 1. / np.sum(np.square(weights))
    
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))

#Compute weighted mean and covariance (per dimension I guess) to get a final state estimate
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    pos = particles[:, :]
    mean = np.average(pos, weights=weights, axis=0)
    bearing = mean[3:6]
    mean[3:6] = bearing / np.linalg.norm(bearing)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def compare_est_pos(est, pos):
    print('State estimate:' + str(est[0]))
    print('Actual position:' + str(pos))    
    print('Absolute Error:' + str(abs(pos - est[0]))) 
    print('Position Error:' + str(sqrt(sum(pos[0:3] - est[0][0:3])**2)))
    print('Dist Err (all var):' + str(sqrt( sum((pos - est[0])**2))))

#scene.forward = vp.vector(0,-1,0)
#scene.camera.pos = vp.vector(0.5,5.5,6.5)
scene.forward = vp.vector(0,-1,0)
scene.camera.pos = vp.vector(0.5,6.5,6.5) #position
#Theta should stay in -pi to pi, phi from 0 to pi
scene.camera.axis = vp.vector(0,-1,0)
scene.up = vp.vector(0,0,-1)
scene.userspin, scene.userpan, scene.userzoom = False, False, False

pos = np.array([scene.camera.pos.x,scene.camera.pos.y,scene.camera.pos.z, scene.camera.axis.norm().x,scene.camera.axis.norm().y,scene.camera.axis.norm().z]) #x y z x bearing y bearing z bearing

#create particles in a cloud of +-3 in each direction within the legal ranges of theta and phi
N = 1000
particles = create_uniform_particles( (pos[0]-3,pos[0]+3) , (pos[1]-3,pos[1]+3), (pos[2]-3,pos[2]+3), (pos[3]-1,pos[3]+1), (pos[4]-1,pos[4]+1), (pos[5]-1,pos[5]+1), N)
threshold = 2*N/3
sensor_std_error = 0.1 #guess
weights = np.ones(N) / N
running = True



def anim_thread(fps = 60):
    dt = 1/fps
    move_error = 0.05 #update this value
    turn_error = 0.05 #update this value
    move_vel = 0.8
    turn_vel = 1
    while running:
        vp.rate(fps)
        k = vp.keysdown()
        u = np.array([0.,0.,0.,0.,0.,0.])
        u = u.astype('float64')
        if 'w' in k:
            #print(scene.forward.norm()*move_vel*dt*(1+move_error*randn()))
            delta = scene.forward.norm()*move_vel*dt*(1+move_error*randn())
            u += np.array([delta.x,delta.y,delta.z,0.,0.,0.])
            scene.camera.pos += delta
        if 's' in k:
            delta = scene.forward.norm()*move_vel*dt*(1+move_error*randn())
            u += np.array([delta.x,delta.y,delta.z,0.,0.,0.])
            scene.camera.pos -= delta
        if 'd' in k:
            delta = scene.forward.cross(scene.up).norm()*move_vel*dt*(1+move_error*randn())
            u += np.array([delta.x,delta.y,delta.z,0.,0.,0.])
            scene.camera.pos += delta
        if 'a' in k:
            delta = scene.forward.cross(scene.up).norm()*move_vel*dt*(1+move_error*randn())
            u += np.array([delta.x,delta.y,delta.z,0.,0.,0.])
            scene.camera.pos -= delta
        if 'up' in k:
            epsilon = scene.camera.axis.rotate(angle = turn_vel*dt*(1+turn_error*randn()), axis = scene.camera.axis.cross(scene.up)).norm()
            delta = (epsilon - scene.camera.axis.norm()).norm()
            u += np.array([0.,0.,0.,delta.x,delta.y,delta.z])
            scene.camera.axis = epsilon
        if 'down' in k:
            epsilon = scene.camera.axis.rotate(angle = -1*turn_vel*dt*(1+turn_error*randn()), axis = scene.camera.axis.cross(scene.up)).norm()
            delta = (epsilon - scene.camera.axis.norm()).norm()
            u += np.array([0.,0.,0.,delta.x,delta.y,delta.z])
            scene.camera.axis = epsilon
        if 'right' in k:          
            epsilon = scene.camera.axis.rotate(angle = -1*turn_vel*dt*(1+turn_error*randn()),axis = scene.up).norm()
            delta = (epsilon - scene.camera.axis.norm()).norm()
            u += np.array([0.,0.,0.,delta.x,delta.y,delta.z])
            scene.camera.axis = epsilon
        if 'left' in k:
            epsilon = scene.camera.axis.rotate(angle = turn_vel*dt*(1+turn_error*randn()),axis = scene.up).norm()
            delta = (epsilon - scene.camera.axis).norm()
            u += np.array([0.,0.,0.,delta.x,delta.y,delta.z])
            scene.camera.axis = epsilon
        scene.camera.axis.mag = 1
        scene.forward.mag = 1
        scene.up.mag = 1
        #Update Pos - do I need error in this?
        pos = np.array([scene.camera.pos.x,scene.camera.pos.y,scene.camera.pos.z, scene.camera.axis.norm().x,scene.camera.axis.norm().y,scene.camera.axis.norm().z]) #x y z x bearing y bearing z bearing
        #Predict
        predict(particles = particles, u = u, std = [move_error,move_error,move_error,turn_error,turn_error,turn_error], dt = 1.)
        #update
        update(particles = particles, weights = weights, z = pos, R = [move_error,move_error,move_error,turn_error,turn_error,turn_error])
        #Neff and resample
        if neff(weights) < threshold:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        #print out error maybe
        est = estimate(particles, weights)
        compare_est_pos(est = est, pos = pos)
        
def pyplot_thread(fps):
    while running:
        est = estimate(particles,weights)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Position tracking of camera')
        ax1 = fig.add_subplot(1,2,1, projection='3d')
        ax1.scatter(particles[:,0], particles[:,1], particles[:,2], color = 'b', marker = 'o')
        ax1.scatter(pos[0],pos[1],pos[2],color = 'r')
        ax1.scatter(est[0][0],est[0][1],est[0][2], color = 'g', marker = '+', linewidth = 5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Cartesian Coordinates')
        ax2 = fig.add_subplot(1,2,2, projection='3d')
        ax2.scatter(particles[:,3], particles[:,4], particles[:, 5], color = 'b', marker = 'o')
        ax2.scatter(pos[3],pos[4], pos[5],color = 'r')
        ax2.scatter(est[0][3],est[0][4],est[0][5], color = 'g', marker = '+', linewidth = 5)
        ax2.set_xlabel('Bearing X')
        ax2.set_ylabel('Bearing Y')
        ax2.set_zlabel('Bearing Z')
        ax2.set_title('3D Bearing')
        plt.show()
        time.sleep(1/fps)

glowscript = threading.Thread(target = anim_thread, args = [60])
glowscript.start()
plot = threading.Thread(target = pyplot_thread, args = [60])
plot.start()


while running:
    command = input('>> ')
    if str.casefold(command) is "quit":
        running = False
        plot.exit()
        glowscript.exit()
    elif str.casefold(command) is "error":
        #TODO add estimation, command line print
        pass
    elif str.casefold(command) is "pos":
        print('Cartesian Coordinates: ' + str(pos[0:2]) + ', Theta: ' + str(pos[3]) + ', Phi: ' + str(pos[4]))
    