#creates video from raw position data

import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

max_fish_count = 30

Xall = np.array([])
Yall = np.array([])

fig = plt.figure(1)
fig.set_size_inches(38.4, 21.6, True)
#fig.set_size_inches(19.2, 10.8, True)
dpi=100
ax = plt.axes(xlim=(6.5, 25), ylim=(3.5, 19.5))

line = ax.scatter([], [], animated=True)

def load_file(fish_filename):

	global X, Y, X_copy, Y_copy
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]

def init():

	global Xall, Yall
	
	npz = np.load("./posAll.npz")
	
	Xall = npz["Xall"]
	Yall = npz["Yall"]
	
	print("Done " + str(Xall.shape))
	
	Xall_new = Xall.flatten()
	Yall_new = Yall.flatten()
	
	print("X box: "+str(np.min( Xall_new[np.isfinite(Xall_new)] ))+" "+str(np.max( Xall_new[np.isfinite(Xall_new)] )) )
	print("Y box: "+str(np.min( Yall_new[np.isfinite(Yall_new)] ))+" "+str(np.max( Yall_new[np.isfinite(Yall_new)] )) )
	
	return line,

def init2():
	line.set_offsets([[],[]])
	return line,

def animate(frame):
	
	global Xall, Yall
	
	pos_all = np.zeros((max_fish_count, 2))
	#pos_all[:,0] = np.cos(np.linspace(0,frame*np.pi, max_fish_count))
	#pos_all[:,1] = np.sin(np.linspace(0,frame*np.pi, max_fish_count))
	#print(pos_all.shape)
	pos_all[:,0] = Xall[:, frame]
	pos_all[:,1] = Yall[:, frame]
	#line.set_offsets(Xall[:,frame], Yall[:,frame])
	#if frame%20 == 0:
	#	print(pos_all)
	line.set_offsets(pos_all)
	return line,

if __name__ == "__main__":

	#plt.style.use('dark_background')
	
	#load_position_data()
	#init_func = init, 
	anim = FuncAnimation(fig, animate, init_func=init, frames = 3136, interval = 40, blit = True)
	
	anim.save('continuousSineWave.mp4', writer = 'ffmpeg', fps = 25, dpi=100)	
	    