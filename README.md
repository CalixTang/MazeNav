# MazeNav

## Purpose
All python files in this repository are in their own folder or in the same folder as this README. Those in the latter were used for example and demonstration purposes while 
writing the ones in their own folders. Their names may suggest the solutions used to tackle the prompt: given a network of blood vessels and a probe that is in that network,
how can we find out where in that network the probe is? As it might suggest, Gaussians were first looked at for use in Kalman filtering, then particle filters were researched.
I ended up using particle filters in the 2D and 3D mazes. As I was unfamiliar with matplotlib before this project, I learned how to use matplotlib by example. It's used to draw
out particles in relation to the actual "state" (probe's position and/or bearing) in the mazes that use particle filtering. In the end, I used an actual graph (nodes and edges)
to model the blood vessel network and used a different kind of filtering to solve the remaining issue at hand: we have no sense of absolute position (which a particle filter 
relies on). 