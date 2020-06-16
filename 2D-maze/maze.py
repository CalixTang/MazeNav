import pyglet as pyg
import random

class Maze:
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.genGrid(height, width)
        
    def genGrid(self,height, width):
        self.grid = [[Cell(pos = [j,i], walls = [True,True,True,True], cellH = 30) for i in range(width)] for j in range(height)]
        self.grid[0][0].walls[0] = False #entrance
        self.grid[-1][-1].walls[2] = False #exit


    def draw(self):
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j].draw()
    
    def genMaze(grid, start, end):
        print(start)
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

maze = Maze(height = 25, width = 25)
Maze.genMaze(grid = maze.grid, start = [0,0], end = [24,24])
window = pyg.window.Window(width = 1200, height = 1000)
label = pyg.text.Label('Maze Gen Alpha',font_name = 'Times New Roman', font_size = 36, x = window.width//2, y = 19*window.height//20, anchor_x = 'center', anchor_y = 'center')
buttonLabel = pyg.text.Label('Maze',font_size = 16, x = 100, y = 925, anchor_x = 'center', anchor_y = 'center')

@window.event
def on_draw():
    window.clear()
    label.draw()
    maze.draw()
    pyg.graphics.draw_indexed(4,pyg.gl.GL_QUADS,[0,1,2,3,0],('v2i', (0,1000,200,1000,200,850,0,850)),('c3B',(120,120,120,120,120,120,120,120,120,120,120,120)))
    buttonLabel.draw()

@window.event
def on_mouse_press(x, y, button, modifiers):
    if x > 0 and x < 200 and y > 850 and y < 1000 and button == pyg.window.mouse.LEFT:
        maze.genGrid(height = len(maze.grid), width = len(maze.grid[0]))
        Maze.genMaze(grid = maze.grid, start = [0,0], end = [24,24])

#Main
pyg.app.run()