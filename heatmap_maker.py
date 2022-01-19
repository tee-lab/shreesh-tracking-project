import matplotlib.pyplot as plt
import numpy as np
import math

#declare arrays that hold X and Y trajectory data
Xall = 1
Yall = 1

X = 1
Y = 1

def load_file(fish_filename):
	
	"""Loads .npz file outputted by Trex and extracts X and Y data"""

	global X, Y
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]

def generate_heatmap():
	
	"""Bins trajectories into a 2D grid and generates a heatmap image"""
	
	global Xall, Yall
	
	#remove all infinities from the array, useful to find true maximum
	Xall_fin = Xall[ np.isfinite(Xall).astype(bool) ]
	Yall_fin = Yall[ np.isfinite(Yall).astype(bool) ]
	
	#first create a bounding box within which the heatmap will be generated
	minx = np.nanmin(Xall_fin)
	maxx = np.nanmax(Xall_fin)
	miny = np.nanmin(Yall_fin)
	maxy = np.nanmax(Yall_fin)
	
	xwidth = maxx-minx
	ywidth = maxy-miny
	
	min_grid_count = 30
	
	if xwidth >= ywidth:
		grid_count_y = min_grid_count
		grid_count_x = math.ceil(grid_count_y*xwidth/ywidth)
	else:
		grid_count_x = min_grid_count
		grid_count_y = math.ceil(grid_count_x*ywidth/xwidth)
	
	#trajectories will be binned into boxes, this array represents the bin frequencies
	frequencies = np.zeros((grid_count_x, grid_count_y), dtype=int)
	
	for fish in range(Xall.shape[0]):
		for frame in range(Xall.shape[1]):
			
			#do not process inifinities
			if Xall[fish, frame] == np.inf:
				continue
			
			#binning step
			x_index = int(math.floor( (Xall[fish, frame] - minx) * grid_count_x / xwidth ))
			y_index = int(math.floor( (Yall[fish, frame] - miny) * grid_count_y / ywidth ))
			
			if x_index == grid_count_x:
				x_index -= 1
			
			if y_index == grid_count_y:
				y_index -= 1
			
			frequencies[x_index, y_index] += 1
	
	print("Ratio of number of blocks in heatmap: " + str(grid_count_x/grid_count_y) + " or " + str(grid_count_y/grid_count_x))
	print("Actual ratio of sides of heatmap: " + str(xwidth/ywidth) + " or " + str(ywidth/xwidth))
	
	plt.imshow(frequencies, cmap='hot', interpolation='gaussian')

if __name__ == "__main__":
	#set this as the path to the directory holding the .npz files
	abs_path = "/home/shreesh/Videos/data/"
	
	#set this to your file name
	video_filename = "30_fish.MOV"

	#set this to your preference
	max_fish_count = 30
	
	#add trajectories to the Xall and Yall arrays
	for count in range(max_fish_count):
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		load_file(fish_filename)
		
		if count == 0:
			Xall = np.array([X])
			Yall = np.array([Y])
		else:
			Xall = np.append(Xall, [X[0:Xall.shape[1]]], axis=0)
			Yall = np.append(Yall, [Y[0:Xall.shape[1]]], axis=0)
	
	generate_heatmap()
	
	plt.show()