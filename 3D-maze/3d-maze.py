import vpython as vp
import random

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
            print(str(i) + " " + str(j) + " " + str(k))
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

#scene.forward = vp.vector(0,-1,0)
#scene.camera.pos = vp.vector(0.5,5.5,6.5)
scene.forward = vp.vector(0,-1,0)
scene.camera.pos = vp.vector(0.5,6.5,6.5)
scene.camera.axis = vp.vector(0,-1,0)
scene.up = vp.vector(0,0,-1)


while True:
    vp.rate(5)
    k = vp.keysdown()
    if 'w' in k:
        scene.camera.pos += scene.forward
    if 's' in k:
        scene.camera.pos -= scene.forward
    if 'up' in k:
        temp = vp.vector(scene.forward.x,scene.forward.y,scene.forward.z)
        scene.forward = scene.up
        scene.camera.axis.mag = 1
        scene.up = -1*temp
    if 'down' in k:
        temp = vp.vector(scene.forward.x,scene.forward.y,scene.forward.z)
        scene.forward = -1*scene.up
        scene.camera.axis.mag = 1
        scene.up = temp
    if 'right' in k:   
        temp = vp.vector(scene.forward.x,scene.forward.y,scene.forward.z)
        scene.forward = -1*vp.vector.cross(scene.up,temp)
        scene.camera.axis.mag = 1
    if 'left' in k:
        temp = vp.vector(scene.forward.x,scene.forward.y,scene.forward.z)
        scene.forward = vp.vector.cross(scene.up,temp)
        scene.camera.axis.mag = 1
