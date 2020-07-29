#3D Maze

##Background
The 3D-maze was the second model I used to simulate a network of blood vessels. I used Wilson's algorithm to generate this maze and the 
vpython library to draw it. The probe is represented by your camera this time and is controlled with wasd as well as arrow keys. Instead of 
move upon release, this maze runs continuously. Every dt is another update of the particle filter. After a while, it should converge. This
filter tracks absolute position (xyz) as well as the probe's bearing (a unit vector pointing where the camera looks). The matplotlib window
has two figures: one showing the particles with probe's position and another showing particles representing the probe's 3D bearing.

##How to Use
When you run 3d-maze.py, a new tab in your main browser will open and run the maze drawing while a matplotlib window opens to show you the particles
and state. This is done with multithreading. Sadly, the program is not yet thread-safe.
You have to close the matplotlib window to update it.
Controls: (first-person)
> Move by holding a key
> wasd keys for translation: w is forward, s is backwards, a is left, d is right.
> arrow keys for rotation: up to look up, down to look down, left to turn left, right to turn right.
> You *can* phase through walls.
I suggest that you run maze.py from the command line, as all printouts of error and position are done there. 
Format for this maze is [x y z x_bearing y_bearing z_bearing].
It's also a continuous stream, so if you want to really look at error you may have to pause the program 
(by clicking on text in the command line). To un-pause, press enter and the stream of text should start flowing again.
If you are paused, you will not be able to move the camera.