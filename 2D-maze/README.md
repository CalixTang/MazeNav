#2D-maze

##Background
The 2D-maze was the first model I used to simulate a network of blood vessels. Grid mazes are simple to create and construct, after all. 
I used Wilson's algorithm to generate this maze and the pyglet library to draw it. The probe is represented by a red ball and is controlled
directionally with the arrow keys. The ball only moves upon key release, so you cannot hold down arrow keys. Also note that you cannot phase 
through walls. Every time you move, the particle filter updates for 50 iterations. After a while, it should converge. You can see the before-
and-after by clicking on "Show PyPlot" in the top right corner. If you click it before your first move, the particle cloud (blue dots) should
be focused around the probe (green x) but not converged totally. After a few moves, the filter should converge to almost being one dot on the 
probe's position.

##How to Use
When you run maze.py, you should see a 'maze.py' window pop up. From here, you can generate a new maze, view the pyplot (matplotlib), or move.
If you open a matplotlib window, you have to close it before continuing to move the probe.
Controls:
> Move upon release
> Directional: up for up, down for down, etc.
> Cannot phase through walls.
Warning: The filter will break if you exit the maze. It also does not move the particle to the probe if you generate a new maze.
I suggest that you run maze.py from the command line, as all printouts of error and position are done there. For this maze, it is [x y].