# Graph

## Background
While the 2D and 3D mazes are good for tracking position, the particle filter that runs them has to have a good guess at initial position
to converge well. That makes less usable solutions for a problem where absolute position or bearing may not be known. This filter tries
to remedy that by modeling the blood vessels as a graph. Instead of having walls that you can or cannot phase through, we have a structure
of nodes, representing the intersections between blood vessels, and edges, representing the vessels between intersections. This filter 
eliminates impossible positions based on the "neighbor" count of each node (the nodes immediately connected to a given node by one edge).
It has no idea of absolute position or where the probe is going, but tries to use deduction to figure out where in the large network the
probe could be.
 
## Usage
When you run graph.py, only a matplotlib window should show up. On the left graph is the full network of blood vessels (randomly generated)
in graph form. The circles are nodes and the rainbow lines are edges. On the right is what the probe "sees". It can "see" the node it is 
currently at and all its neighbors. It also "knows" the full history of where it has been and seen. This means the nodes and edges on the right
will accumulate as you move the probe around the graph/network.

This graph support history backtracking and re-tracking. You can use left and right arrow keys to go back/forward one node in time. This will 
bring the weights back or forward one node in time as well. The small graph (H) will not change. If you are back in time and make a new move, 
the moves AND weights that come after where you were in history *be deleted*. Because the small graph doesn't change to accomodate this,
the filter's output can be messed up. If you are back and re-tracking, stick with only using the arrow keys.

Reading the graphs:
>The numbers on the nodes are ID numbers. For the sake of a good filter, the left and right ID numbers are different.
>The red ball is the probe.
>A black node outline symbolizes a neighbor node of the probe. 
>A green outline symbolizes a possibility for the probe's position. On start, every node should be green.
>As you move the probe around, possibilities should narrow down.

Controls:
>Left click on a node in the left graph to move to it. This will *only move the probe to that node if it's a neighbor of the current node.*
>Middle click on a node in the left graph to forcefully move the probe. This will break the filter, so I don't suggest you use this.
>Right click on a node in the left graph to get a command-line printout of all its neighbors. This is an outdated feature.
>Enter to open a new figure that displays a bar plot of all nodes and weights.
>Left Arrow to go back one node of history.
>Right Arrow to go back one node of history.

I suggest that you run maze.py from the command line, as all printouts of possible node positions are done there. The printout is a sorted array
from high to low of all nodes and their possibilities of being where the probe is at.