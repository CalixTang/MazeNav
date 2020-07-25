import pyglet as pyg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import *
import random

wwidth = 1200
wheight = 1000
'''window = pyg.window.Window(width = wwidth, height = wheight)
pyg.gl.glClearColor(0.5,0,0,1)
R,G,B,A = 120,120,120,255
bckgrd = pyg.image.SolidColorImagePattern((R,G,B,A)).create_image(wwidth,wheight)'''

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


   
#Ball: player controlled
class Ball:
    def __init__(self, radius, pos, id, bcolor):
        self.radius = radius
        self.pos = pos
        self.id = id
        self.bcolor = bcolor
    def draw(self):
        makeCircle(100,self.radius,self.pos,color=self.bcolor).draw(pyg.gl.GL_LINE_LOOP)
        pyg.text.Label(text = str(self.id), font_name = 'Arial', font_size = self.radius, x = self.pos[0], y = self.pos[1], anchor_x = 'center', anchor_y = 'center').draw()


def fillGraph(N):
    pos = -1*np.ones((N,2))
    for i in range(N):
        xpos, ypos = np.random.rand(), np.random.rand()
        for j in range(i):
            if np.linalg.norm(pos[j] - np.array([xpos,ypos])) > 2*node_radius_mpl:
                continue
            else:
                xpos, ypos = np.random.rand(), np.random.rand()
                j -= j
        G.add_node(i, x = xpos, y = ypos)
        pos[i] = np.array([xpos,ypos])
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
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    id = -1
    for i in range(n):
        if sqrt((G.nodes.data('x')[i] - event.xdata)**2 + (G.nodes.data('y')[i] - event.ydata)**2) <= node_radius_mpl:
            id = i
            #print('Clicked Node: ' + str(id))
    if id > 0 and id in G.adj[ball.id] and event.button == 1:
        ball.id = id
        plt.clf()
        plot()
    elif id > 0 and event.button == 2:
        ball.id = id
        plt.clf()
        plot()
    elif id > 0 and event.button == 3:
        print('Adjacent nodes: ' + str(G.adj[id]))
    fig.canvas.draw()

def plot():
    axs[0].set_title('Interactive graph')
    axs[1].set_title('Camera\'s vision tracker')
    axs[0].set_xlim(0,1)
    axs[0].set_ylim(0,1)
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)
    #interactive graph
    axs[0].scatter(np.array(xs)[:,1],np.array(ys)[:,1], marker = 'o', color = '#66b3ff', linewidth = 1, s = 300)
    for (i, u) in G.nodes.data():
        axs[0].text(x = u['x'], y = u['y'], s = str(i), color = 'w', horizontalalignment = 'center', verticalalignment = 'center')
    axs[0].scatter(xs[ball.id],ys[ball.id], marker = 'o', edgecolor = 'k', color = 'r', linewidth = 1, s = 100)
    for (u, v, w) in G.edges.data():
        axs[0].plot([xs[u],xs[v]],[ys[u],ys[v]], linewidth = 0.5)
    
    #vision graph
    axs[1].scatter(np.array(vxs)[:,1],np.array(vys)[:,1], marker = 'o', edgecolor = 'k', color = '#66b3ff', linewidth = 1, s = 300)
    for (i, u) in H.nodes.data():
        axs[1].text(x = u['x'], y = u['y'], s = str(i), color = 'w', horizontalalignment = 'center', verticalalignment = 'center')
    #axs[1].scatter(vxs[ball.id],vys[ball.id], marker = 'o', edgecolor = 'k', color = 'r', linewidth = 1, s = 100)
    for (u, v, w) in H.edges.data():
        axs[1].plot([vxs[u],vxs[v]],[vys[u],vys[v]], linewidth = 0.5, color = 'k')
    
n = 50    
ball = Ball(12, [800, 100], random.randint(0,n-1), (255,0,0)) #human controlled   
node_radius_mpl = 0.0212956146 #DO NOT TOUCH
G = nx.Graph()
H = nx.Graph()
'''for (i, u) in G.nodes.data():
    print(str(i) + ' ' + str(u))'''
fillGraph(n)
xs, ys = G.nodes.data('x'), G.nodes.data('y')
H.add_node(0, x = xs[ball.id], y = ys[ball.id])
print(G.adj[ball.id])
t0 = 1
for u in G.adj[ball.id]:
    H.add_node(t0, x = xs[u], y = ys[u])
    t0 += 1
    
vxs, vys = H.nodes.data('x'), H.nodes.data('y')
#fig = plt.figure(figsize = (12,8)) #PLEASE DON'T RESIZE. THE NODE MARKERS (CIRCLES) DON'T RESIZE WHEN THE WINDOW DOES AND IT MESSES THINGS UP.
fig, axs = plt.subplots(1, 2, constrained_layout = True, figsize = (12,8))


cid = fig.canvas.mpl_connect('button_press_event', onclick)


plot()
plt.show()
plt.ion()
