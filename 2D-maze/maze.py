import pyglet as pyg
import random
from math import *
from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt 
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats

#a generic makeCircle modified for use here.
def makeCircle(numPoints, radius, pos, color):
    verts = []
    colors = []
    for i in range(numPoints):
        angle = radians(float(i)/numPoints * 360.0)
        x = radius*cos(angle) + pos[0]
        y = radius*sin(angle) + pos[1]
        verts += [x,y]
        colors += color
    return pyg.graphics.vertex_list(numPoints, ('v2f', verts), ('c3B', colors))

class Maze:
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.genGrid(height, width)
        
    def genGrid(self,height, width):
        self.grid = [[Cell(pos = [j,i], walls = [True,True,True,True], cellH = 30) for i in range(width)] for j in range(height)]
        self.grid[0][0].walls[0] = False #entrance
        self.grid[-1][-1].walls[2] = False #exit

    def getCellFromPos(self,pos,radius):
        for row in self.grid:
            for cell in row:
                if cell.contains(pos,radius):
                    return cell
        return None 
    
    def draw(self):
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].draw()
    
    def genMaze(grid, start, end):
        if Maze.checkGrid(grid) == True:
            return
        else:
            grid[end[0]][end[1]].inMaze = True #add the end to the maze
            begin = start[:] #we'll use this during cutting
            #Random walk starting at the start (this should take a while)
            while not grid[start[0]][start[1]].inMaze:
                grid[start[0]][start[1]].dir = Maze.randomWalk([start[0],start[1]], len(grid), len(grid[0]))
                if grid[start[0]][start[1]].dir == 0:
                    start[0] -= 1
                    pass
                elif grid[start[0]][start[1]].dir == 1:
                    start[1] += 1
                    pass
                elif grid[start[0]][start[1]].dir == 2:
                    start[0] += 1
                    pass
                elif grid[start[0]][start[1]].dir == 3:
                    start[1] -= 1
                    pass
            #begin "cutting" a path out starting from the beginning following dir, put each cell that it follows into the maze
            while not begin[0] == end[0] and begin[1] == end[1]:
                grid[begin[0]][begin[1]].walls[grid[begin[0]][begin[1]].dir] = False
                grid[begin[0]][begin[1]].inMaze = True
                if grid[begin[0]][begin[1]].dir == 0:
                    begin[0] -= 1
                    grid[begin[0]][begin[1]].walls[2] = False
                elif grid[begin[0]][begin[1]].dir == 1:
                    begin[1] += 1
                    grid[begin[0]][begin[1]].walls[3] = False
                elif grid[begin[0]][begin[1]].dir == 2:
                    begin[0] +=1
                    grid[begin[0]][begin[1]].walls[0] = False
                elif grid[begin[0]][begin[1]].dir == 3:
                    begin[1] -= 1
                    grid[begin[0]][begin[1]].walls[1] = False
            #call genRandom now because you have a first path! :D
            Maze.genMazeR(grid)

    def genMazeR(grid):
        if Maze.checkGrid(grid) == True:
            return
        else:
            #find a random unMazed cell to start
            x = random.randint(0,len(grid)-1)
            y = random.randint(0,len(grid[0])-1)
            while grid[x][y].inMaze:
                x = random.randint(0,len(grid)-1)
                y = random.randint(0,len(grid[0])-1)
            begin = [x,y]
            #random walk until we hit a cell that's in the maze (yes, x and y are "flipped" here)
            while not grid[x][y].inMaze:
                grid[x][y].dir = Maze.randomWalk([x,y], len(grid), len(grid[0]))
                if grid[x][y].dir == 0:
                    x -= 1
                elif grid[x][y].dir == 1:
                    y += 1
                elif grid[x][y].dir == 2:
                    x += 1
                elif grid[x][y].dir == 3:
                    y -= 1    
            #cut
            while not grid[begin[0]][begin[1]].inMaze:
                grid[begin[0]][begin[1]].walls[grid[begin[0]][begin[1]].dir] = False
                grid[begin[0]][begin[1]].inMaze = True
                if grid[begin[0]][begin[1]].dir == 0:
                    begin[0] -= 1
                    grid[begin[0]][begin[1]].walls[2] = False
                elif grid[begin[0]][begin[1]].dir == 1:
                    begin[1] += 1
                    grid[begin[0]][begin[1]].walls[3] = False
                elif grid[begin[0]][begin[1]].dir == 2:
                    begin[0] +=1
                    grid[begin[0]][begin[1]].walls[0] = False
                elif grid[begin[0]][begin[1]].dir == 3:
                    begin[1] -= 1
                    grid[begin[0]][begin[1]].walls[1] = False
            #recur
            Maze.genMazeR(grid)
    
    #outputs a random direction to go in for a cell input in terms of its position in a 2D array. 
    def randomWalk(pos, height, width):
        t0 = random.randint(0,3)
        #print(str(pos) + ' ' + str(height) + ' ' + str(width) + ' ' + str(t0))
        if (t0 == 0 and pos[0] == 0): 
            return Maze.randomWalk(pos, height, width)
        if (t0 == 1 and pos[1] == width-1):
            return Maze.randomWalk(pos, height, width)
        if (t0 == 2 and pos[0] == height-1): 
            return Maze.randomWalk(pos, height, width)
        if (t0 == 3 and pos[1] == 0):
            return Maze.randomWalk(pos, height, width)
        else:
            return t0
    
    def checkGrid(grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not grid[i][j].inMaze:
                    return False
        return True
    
#A cell is a 1x1 area with four walls around it by default, marked by NESW directions. It has a defined position vector.
class Cell:
    
    
    def __init__(self, pos, walls, cellH):
        self.pos = pos
        self.walls = walls
        #self.visited = False
        self.dir = -1 #-1 0, 1, 2, or 3 when in maze
        self.inMaze = False
        self.cellH = cellH
   
    def draw(self):
        if self.walls[0] == True:
            pyg.graphics.draw_indexed(2,pyg.gl.GL_LINE_STRIP,[0,1],('v2i',(215 + self.cellH*self.pos[1] - self.cellH//2, 785 - self.cellH*self.pos[0] + self.cellH//2, 215 + self.cellH*self.pos[1] + self.cellH//2, 785 - self.cellH*self.pos[0] + self.cellH//2)))
        if self.walls[1] == True:
            pyg.graphics.draw_indexed(2,pyg.gl.GL_LINE_STRIP,[0,1],('v2i',(215 + self.cellH*self.pos[1] + self.cellH//2, 785 - self.cellH*self.pos[0] + self.cellH//2, 215 + self.cellH*self.pos[1] + self.cellH//2, 785 - self.cellH*self.pos[0] - self.cellH//2)))
        if self.walls[2] == True:
            pyg.graphics.draw_indexed(2,pyg.gl.GL_LINE_STRIP,[0,1],('v2i',(215 + self.cellH*self.pos[1] + self.cellH//2, 785 - self.cellH*self.pos[0] - self.cellH//2, 215 + self.cellH*self.pos[1] - self.cellH//2, 785 - self.cellH*self.pos[0] - self.cellH//2)))
        if self.walls[3] == True:
            pyg.graphics.draw_indexed(2,pyg.gl.GL_LINE_STRIP,[0,1],('v2i',(215 + self.cellH*self.pos[1] - self.cellH//2, 785 - self.cellH*self.pos[0] - self.cellH//2, 215 + self.cellH*self.pos[1] - self.cellH//2, 785 - self.cellH*self.pos[0] + self.cellH//2)))

    #contains tells us whether a ball with given radius will fit, given the position of the ball.
    def contains(self,pos,radius):
        #print([abs(215 + self.cellH*self.pos[1] - pos[0]), abs(785 - self.cellH*self.pos[0] - pos[1])])
        if abs(215 + self.cellH*self.pos[1] - pos[0]) <= self.cellH - radius and abs(785 - self.cellH*self.pos[0] - pos[1]) <= self.cellH - radius:
            return True
        else:
            return False

#Ball: player controlled
class Ball:
    def __init__(self, radius, pos):
        self.radius = radius
        self.pos = pos
        
    def draw(self):
        makeCircle(100,self.radius,self.pos,color=(255,0,0)).draw(pyg.gl.GL_LINE_LOOP)

window = pyg.window.Window(width = 1200, height = 1000)

@window.event
def on_mouse_press(x, y, button, modifiers):
    if x > 0 and x < 200 and y > 850 and y < 1000 and button == pyg.window.mouse.LEFT: #gen new maze
        maze.genGrid(height = len(maze.grid), width = len(maze.grid[0]))
        Maze.genMaze(grid = maze.grid, start = [0,0], end = [19,29])
        ball.pos = [215 + maze.grid[0][0].cellH*random.randint(0,len(maze.grid[0])-1), 785 - maze.grid[0][0].cellH*random.randint(0,len(maze.grid)-1)]
        particles = create_uniform_particles([ball.pos[0] - 200,ball.pos[0] + 200],[ball.pos[1] - 200,ball.pos[1] + 200], N)
        weights = np.ones(N)/N
        pos = np.array(ball.pos)
        sensor_std_error = (5,5) #guess
        xs = update_and_resample(particles = particles, weights = weights, iters = 50, threshold = 3*N/4, R = sensor_std_error)
        compare_est_pos()
    if x > 1000 and x < 1200 and y > 850 and y < 1000 and button == pyg.window.mouse.LEFT: #pyplot
        plotPoints()
        

#key controls - arrows to move the ball around, w to get the walls of the current cell.
@window.event
def on_key_release(symbol,modifiers):
    if symbol == pyg.window.key.DOWN:
        currentCell = maze.getCellFromPos(pos = ball.pos, radius = ball.radius)
        if not currentCell.walls[2]:
            ball.pos[1] -= currentCell.cellH
            predict(particles = particles, u = (0,-30), std = (5,5))
            update_and_resample(particles = particles, weights = weights, iters = 50, threshold = 2*N/3, R = sensor_std_error)
            compare_est_pos()
    elif symbol == pyg.window.key.UP:
        currentCell = maze.getCellFromPos(pos = ball.pos, radius = ball.radius)
        if not currentCell.walls[0]:
            ball.pos[1] += currentCell.cellH
            predict(particles = particles, u = (0,30), std = (5,5))
            update_and_resample(particles = particles, weights = weights, iters = 50, threshold = 2*N/3, R = sensor_std_error)
            compare_est_pos()
    elif symbol == pyg.window.key.LEFT:
        currentCell = maze.getCellFromPos(pos = ball.pos, radius = ball.radius)
        if not currentCell.walls[3]:
            ball.pos[0] -= currentCell.cellH
            predict(particles = particles, u = (-30,0), std = (5,5))
            update_and_resample(particles = particles, weights = weights, iters = 50, threshold = 2*N/3, R = sensor_std_error)
            compare_est_pos()
    elif symbol == pyg.window.key.RIGHT:
        currentCell = maze.getCellFromPos(pos = ball.pos, radius = ball.radius)
        if not currentCell.walls[1]:
            ball.pos[0] += currentCell.cellH
            predict(particles = particles, u = (30,0), std = (5,5))
            update_and_resample(particles = particles, weights = weights, iters = 50, threshold = 2*N/3, R = sensor_std_error)
            compare_est_pos()            
    elif symbol == pyg.window.key.W:
        print(maze.getCellFromPos(pos = ball.pos, radius = ball.radius).walls)


@window.event
def on_draw():
    window.clear()
    label.draw()
    maze.draw()
    pyg.graphics.draw_indexed(4,pyg.gl.GL_QUADS,[0,1,2,3,0],('v2i', (0,1000,200,1000,200,850,0,850)),('c3B',(120,120,120,120,120,120,120,120,120,120,120,120)))
    pyg.graphics.draw_indexed(4,pyg.gl.GL_QUADS,[0,1,2,3,0],('v2i', (1000,1000,1200,1000,1200,850,1000,850)),('c3B',(120,120,120,120,120,120,120,120,120,120,120,120)))
    buttonLabel.draw()
    ball.draw()
    plotLabel.draw()
    #for p in particles:
    #    makeCircle(100,3,[p[0],p[1]],color=(0,0,255)).draw(pyg.gl.GL_LINE_LOOP)

#Filtering
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles
    
def predict(particles, u, std, dt=1.):
    """ move according to control input u [dx/dt, dy/dt]
    with noise Q """

    N = len(particles)

    # move in the (noisy) commanded direction
    dist = [(u[0] * dt) + (randn(N) * std[0]),(u[1] * dt) + (randn(N) * std[1])]
    particles[:, 0] += dist[0]
    particles[:, 1] += dist[1]

#Update - update weights based on measurement - I'm using 1 / euclidean distance
def update(particles, weights, z, R):
    """for i, landmark in enumerate(landmarks):
    distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
    weights *= scipy.stats.norm(distance, R).pdf(z[i])"""

    weights = 1 / (np.sqrt( (particles[:,0] - z[0])**2 + (particles[:,1] - z[1])**2 ) + 1.e-100) # 1/(euclidean + epsilon)
    weights /= sum(weights) # normalize
    return weights
    
#Resample - discard low probability particles and dupe high ones
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)
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
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var
    
#update and resample at the same time
def update_and_resample(particles, weights, iters, threshold, R):
    xs = []
    for i in range(iters):
        weights = update(particles = particles, weights = weights, z = ball.pos, R = R)
        # resample if too few effective particles
        #print(weights)        
        #print(neff(weights))
        if neff(weights) < threshold:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights)
        xs.append( [mu,var] )
    return xs
def compare_est_pos():
    e = estimate(particles,weights)[0]
    print('State estimate:' + str(e))
    print('Ball position :' + str(ball.pos))    
    print('Absolute Error:' + str(abs(ball.pos - e))) 
    print('Distance Error:' + str(sqrt( sum((ball.pos - e)**2))))


#Main
maze = Maze(height = 20, width = 30)
Maze.genMaze(grid = maze.grid, start = [0,0], end = [19,29])

label = pyg.text.Label('Maze Gen Alpha',font_name = 'Times New Roman', font_size = 36, x = window.width//2, y = 19*window.height//20, anchor_x = 'center', anchor_y = 'center')
buttonLabel = pyg.text.Label('Maze',font_size = 16, x = 100, y = 925, anchor_x = 'center', anchor_y = 'center')
plotLabel = pyg.text.Label('Show PyPlot',font_size = 16, x = 1100, y = 925, anchor_x = 'center', anchor_y = 'center')
ball = Ball(6, [215 + maze.grid[0][0].cellH*random.randint(0,len(maze.grid[0])-1), 785 - maze.grid[0][0].cellH*random.randint(0,len(maze.grid)-1)])

#I'm going to generate particles randomly near the ball's position, say +- 200 pixels
N = 1000
particles = create_uniform_particles([ball.pos[0] - 200,ball.pos[0] + 200],[ball.pos[1] - 200,ball.pos[1] + 200], N)
weights = np.ones(N)/N
pos = np.array(ball.pos)
sensor_std_error = (5,5) #guess
xs = update_and_resample(particles = particles, weights = weights, iters = 50, threshold = 3*N/4, R = sensor_std_error)
compare_est_pos()



def plotPoints():
    plt.scatter(particles[:,0],particles[:,1],color = 'b',marker = 'o')
    plt.scatter(ball.pos[0],ball.pos[1],color = 'g', marker = 'x')
    plt.xlim(0,1200)
    plt.ylim(0,1000)
    plt.plot([200,1100,1100,200,200],[800,800,200,200,800],color = 'b')
    plt.show()
    plt.draw()

pyg.app.run()
