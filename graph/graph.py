import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample
from math import *
import random


#Ball: player controlled
class Ball:
    def __init__(self, pos, id):
        self.pos = pos #where ball is in H
        self.id = id #where ball is in G
    

def fillGraph(N): #generates random-ish graph with nodes 0 to n-1
    pos = -1*np.ones((N,2))
    
    for i in range(N):
        xpos, ypos = np.random.rand(), np.random.rand()
        j = 0
        while j < i:
            if sqrt((pos[j][0]-xpos)**2 + (pos[j][1]-ypos)**2) > 2.5*node_radius_mpl:
                j += 1
            else:
                xpos, ypos = np.random.rand(), np.random.rand()
                j = 0
        G.add_node(i, x = xpos, y = ypos)
        pos[i][0], pos[i][1] = xpos,ypos
        #print(pos)
    for i in range(N):
        if N > 20:
            a = random.randint(1,int(N/20))
            for j in range(a):
                b = random.randint(0,N-1)
                while b is i:
                    b = random.randint(0,N)
                G.add_edges_from([(b,i)])
        else:        
            a = random.randint(0,int(N/5))
            for j in range(a):
                b = random.randint(0,N-1)
                while b is i:
                    b = random.randint(0,N)
                G.add_edges_from([(b,i)])

def onclick(event):
    '''print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))'''
    id = -1
    if event.xdata is not None and event.ydata is not None: 
        for i in range(n):
            if sqrt((G.nodes.data('x')[i] - event.xdata)**2 + (G.nodes.data('y')[i] - event.ydata)**2) <= node_radius_mpl:
                id = i
                #print('Clicked Node: ' + str(id))
        if id > -1 and id in G.adj[ball.id] and event.button == 1:
            global vxs
            global vys
            global prev_moves
            global prev_weights
            global weights
            global hist_index
            ball.id = id #update the ball to be in the adjacent node.
            
            #find the corresponding H node for the G node you move to.
            t0 = 0
            dist = 1.0e30
            for i in range(H.number_of_nodes()):
                d = sqrt((vxs[i]-xs[ball.id])**2 + (vys[i]-ys[ball.id])**2)
                if d < dist:
                    t0 = i
                    dist = d
            ball.pos = t0
           
            
            
            for u, v in G.adj[ball.id].items(): #for all neighbors in G
                #if the neighbor is in H already, check if an edge exists
                index = -1
                for i in H.nodes:
                    if vxs[i] == xs[u] and vys[i] == ys[u]:
                        index = i
                        break
                if index > -1:
                    if (ball.pos,index) not in H.edges and (index, ball.pos) not in H.edges:
                        H.add_edges_from([(ball.pos,index)])
                #Otherwise just add a new node
                else:
                    H.add_node(H.number_of_nodes(), x = xs[u], y = ys[u]) #if the adjacent's x and y is not in H, add it and increment t0
                    H.add_edges_from([(ball.pos,H.number_of_nodes()-1)])     
            
            vxs, vys = H.nodes.data('x'), H.nodes.data('y')
            #filtering time
            #propagate
            weights = propagate(weights)
            #update
            weights = update(weights, ball.pos)
            #estimate
            estimate(weights)
            
            prev_moves = np.hstack((prev_moves[:,0:hist_index],np.array([ball.id,ball.pos]).reshape(2,1)))
            prev_weights = np.hstack((prev_weights[:,0:hist_index], weights.reshape(n,1)))
            hist_index = len(prev_weights[0])
            
            for ax in axs:
                ax.clear()
            plot()
        elif id > 0 and event.button == 2:
            ball.id = id
            for ax in axs:
                ax.clear()
            plot()
        elif id > 0 and event.button == 3:
            print(str(len(G.adj[id].items())) + ' neighbors.')
            print(str(G.adj[id].items()))
        fig.canvas.draw()
    else:
        pass

def press(event):
    global hist_index
    global weights
    if event.key == "enter":
        plot_prob(weights)
    elif event.key == "left":
        if hist_index > 0:
            hist_index -= 1
            ball.id, ball.pos = prev_moves[0][hist_index-1], prev_moves[1][hist_index-1]
            weights = prev_weights[:,hist_index-1]
            for ax in axs:
                ax.clear()
            plot()
            fig.canvas.draw()
        else:
            print("Can't go any further back in time!")
    elif event.key == "right":
        if hist_index < len(prev_weights[0]):
            hist_index += 1
            ball.id,ball.pos = prev_moves[0][hist_index-1], prev_moves[1][hist_index-1]
            weights = prev_weights[:,hist_index-1]
            for ax in axs:
                ax.clear()
            plot()
            fig.canvas.draw()
        else:
            print("Can't go any more forward in time!")
    
def plot():
    axs[0].set_title('Interactive graph')
    axs[1].set_title('Camera\'s vision tracker')
    axs[0].set_xlim(0,1)
    axs[0].set_ylim(0,1)
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)
    
    #interactive graph
    #draw edges
    for (u, v, w) in G.edges.data():
        axs[0].plot([xs[u],xs[v]],[ys[u],ys[v]], linewidth = 0.5)
    
    #draw nodes
    axs[0].scatter(np.array(xs)[:,1],np.array(ys)[:,1], marker = 'o', color = '#66b3ff', linewidth = 1, s = 300)
    #draw possible probe positions and give them a larger, green outline
    global weights
    for i in range(len(weights)):
        if weights[i] > 0:
            axs[0].scatter(xs[i],ys[i], marker = 'o', edgecolor = 'g', color = '#66b3ff', linewidth = 2, s = 420)
    #draw neighbors and give them a black outline
    for u in G.adj[ball.id]:
        axs[0].scatter(xs[u],ys[u], marker = 'o', edgecolor = 'k', color = '#66b3ff', linewidth = 2, s = 300)
    
    #put ID on each node
    for (i, u) in G.nodes.data():
        axs[0].text(x = u['x'], y = u['y'], s = str(i), color = 'w', horizontalalignment = 'center', verticalalignment = 'center')
    #draw the probe
    axs[0].scatter(xs[ball.id],ys[ball.id], marker = 'o', edgecolor = 'k', color = 'r', linewidth = 1, s = 100)
    
    #vision graph
    vxs, vys = H.nodes.data('x'), H.nodes.data('y')
    axs[1].scatter(np.array(vxs)[:,1],np.array(vys)[:,1], marker = 'o', edgecolor = 'k', color = '#66b3ff', linewidth = 1, s = 300)
    for (i, u) in H.nodes.data():
        axs[1].text(x = u['x'], y = u['y'], s = str(i), color = 'w', horizontalalignment = 'center', verticalalignment = 'center')
    axs[1].scatter(xs[ball.id],ys[ball.id], marker = 'o', edgecolor = 'k', color = 'r', linewidth = 1, s = 100)
    for (u, v, w) in H.edges.data():
        axs[1].plot([vxs[u],vxs[v]],[vys[u],vys[v]], linewidth = 0.5, color = 'k')

def plot_prob(weights):
    width = 0.8
    fig, ax = plt.subplots()
    xs = np.arange(len(weights))
    ax.set_xlabel('Node Number')
    ax.set_ylabel('Probabilities')
    ax.set_xticks(xs)
    ax.set_ylim(bottom = 0.0, top = 1.0)
    ax.bar(xs, weights, width)
    fig.show()

def propagate(weights):
    new = np.zeros(G.number_of_nodes())
    for i in range(G.number_of_nodes()): #for each node
        for (j, u) in G.adj[i].items():
            new[j] += weights[i] / len(G.adj[i].items())
    new /= sum(new)
    #print(new.reshape(5,10))
    return new
    
def update(weights, move):
    '''print(len(H.adj[move].items()))
    print(H.adj[move].items())'''
    prob = weights[:]
    #Determine immediate possibility for the particle to be at a node based on neighbors.
    for i in range(len(weights)):
        neighbors = G.adj[i].items()
        if len(neighbors) != len(H.adj[move].items()):#If impossible: penalize.
            #prob[i] = 0
            pass
        else: #If possible: weight = average of neighbors?
            #prob[i] *= 2
            prob[i] *= 5
            '''for (j, u) in neighbors:
                prob[i] += weights[j] 
            prob[i] /= len(neighbors)'''   
    prob /= sum(prob)
    return prob
    
def estimate(weights):    
    ranks = np.flip(np.argsort(weights))
    print([str(ranks[i]) + ': ' + str(weights[ranks[i]]) for i in range(len(weights))])
   

n = 50    
ball = Ball(0, random.randint(0,n-1)) #human controlled   

node_radius_mpl = 0.0232956146 #DO NOT TOUCH
G = nx.Graph()
H = nx.Graph()
weights = np.ones(n) / n
'''for (i, u) in G.nodes.data():
    print(str(i) + ' ' + str(u))'''
fillGraph(n)
xs, ys = G.nodes.data('x'), G.nodes.data('y')
H.add_node(0, x = xs[ball.id], y = ys[ball.id])
for u in G.adj[ball.id]:
    H.add_node(H.number_of_nodes(), x = xs[u], y = ys[u])
    H.add_edges_from([(0,H.number_of_nodes()-1)])
    
prev_moves = np.array([ball.id,0]).reshape(2,1)
prev_weights = weights[:].reshape(n,1)
print(prev_weights)
hist_index = 1
    
vxs, vys = H.nodes.data('x'), H.nodes.data('y')
#fig = plt.figure(figsize = (12,8)) #PLEASE DON'T RESIZE. THE NODE MARKERS (CIRCLES) DON'T RESIZE WHEN THE WINDOW DOES AND IT MESSES THINGS UP.
fig, axs = plt.subplots(1, 2, constrained_layout = True, figsize = (12,8))


cid = fig.canvas.mpl_connect('button_press_event', onclick)
cidtwo = fig.canvas.mpl_connect('key_press_event', press)

plot()
plt.show()
plt.ion()
