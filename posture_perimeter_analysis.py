import sys
import numpy as np
from random import random
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
#from scipy.cluster.hierarchy import fclusterdata

posture_frames = np.array(1)
outlines = np.array(1)
midlines = np.array(1)
outline_lengths = np.array(1)
midline_lengths = np.array(1)
outline_iterator = 0
midline_iterator = 0
X = np.array(1)
Y = np.array(1)
X_copy = np.array(1)
Y_copy = np.array(1)
fish_count = -1

Xall = np.array([])
Yall = np.array([])
frames_array = []

offsets = np.array(1)

cm_per_pixel = 0.018428 #trex internal parameter value

#the following two defs are utility functions

def moving_average(X, span):
	return np.convolve(X, np.ones(span), 'valid') / span

def get_distance_nparray(points):
	return get_distance(points[0,0], points[0,1], points[1,0], points[1,1])

def get_distance(x1, y1, x2, y2):
	return ( (x2-x1)**2 + (y2-y1)**2 )**0.5

def get_slope(points):
	return (points[1,1] - points[0,1])/(points[1,0] - points[0,0])

def get_nearest(fish_listX, fish_listY, x, y):
	dists = get_distance(fish_listX, fish_listY, x, y)
	return np.argsort(dists)[1]

def find_inactive_fish(fish_list):
	inactive_indices = []
	
	for i in range(len(fish_list)):
		if fish_list[i] == np.inf:
			inactive_indices.append(i)
	
	return inactive_indices

#the following are to load trex data files

def load_posture_file(fish_filename):
	
	global outlines, outline_lengths
	global midlines, midline_lengths
	global offsets, posture_frames
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	outlines = npz["outline_points"]*cm_per_pixel
	midlines = npz["midline_points"]*cm_per_pixel
	
	outline_lengths = npz["outline_lengths"]
	midline_lengths = npz["midline_lengths"]
	
	offsets = npz["offset"]*cm_per_pixel
	
	posture_frames = npz["frames"]

def load_file(fish_filename):

	global X, Y, X_copy, Y_copy, frames_array
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]
	
	X_copy = npz["X#wcentroid"]
	Y_copy = npz["Y#wcentroid"]
	
	if len(frames_array) == 0:
		frames_array = npz["frame"]

def get_perimeter(outline):
	
	"""This function takes in an outline (periodic) and outputs its perimeter."""
	
	#to store periodic version of outline for processing purposes
	outline_new = np.zeros((outline.shape[0]+1,2))
	outline_new[-1,:] = outline[0,:]
	outline_new[0:-1,:] = outline
	
	segment_lengths = get_distance(outline_new[0:-1,0], outline_new[0:-1,1], outline_new[1:,0], outline_new[1:,1])
	
	return sum(segment_lengths)

def find_angles(slopes):
	
	"""This function takes a periodic array of slopes and outputs
	the corresponding array of angles the slopes make with the x axis.
	These angles are continuous, ie they vary from theta->theta +/- pi"""
	
	#first find straightforward arctan
	tan_arr = np.arctan(slopes)
	
	#this will be used to correct discontinuities in the arc tan output
	increment = 0
	
	for i in range(len(tan_arr)):
		
		if tan_arr[i-1]-tan_arr[i] > np.pi*0.5: #if angles shifted down by pi
			increment += np.pi
		elif tan_arr[i-1]-tan_arr[i] < -np.pi*0.5: #if angles shifted up by pi
			increment -= np.pi
		
		tan_arr[i-1] += increment
	
	return tan_arr

def is_point_inside(outline, point):
	
	"""This function checks if the given point is inside the given outline.
	It uses the property that the rays connecting points on the outline with
	the given point will wind around a circle once as we traverse the outline
	iff the point is inside the outline, while if the point is outside, the
	winding amount will be zero."""
	
	slopes = (outline[:,1]-point[1])/(outline[:,0]-point[0])
	slope_angle = find_angles(slopes)
	
	return abs(slope_angle[-2]-slope_angle[0]) > np.pi*0.9

def fill_points(outline):
	
	"""This function takes in an array of points representing a continuous
	and periodic outline of a body and outputs an array of points containting
	the points of the outline along with randomly sampled points inside the outline."""
	
	newpoints = np.zeros((0,2))
	
	#initially populate with only points on the outline
	#newpoints[0:outline.shape[0],:] = outline
	#newpoints[outline.shape[0]:outline.shape[0]*2,:] = outline
	#newpoints[2*outline.shape[0]:3*outline.shape[0],:] = outline
	#newpoints[3*outline.shape[0]:,:] = outline
	
	#used to sample random points inside the bounding box of the outline
	minx = min(outline[:,0])
	miny = min(outline[:,1])
	maxx = max(outline[:,0])
	maxy = max(outline[:,1])
	
	numpoints = 3*int(outline.shape[0]**0.5)
	
	xstep = (maxx-minx)/numpoints
	ystep = (maxy-miny)/numpoints
	
	#do not fill points if non-finite values found
	if minx == -np.inf or miny == -np.inf or maxx == np.inf or maxy == np.inf:
		return newpoints
	
	#for i in range(3*outline.shape[0]):
	for i in range(numpoints):
		newpoint = np.zeros(2)
		newpoint[0] = minx + i*xstep
		for j in range(numpoints):
			
			newpoint[1] = miny + j*ystep
			
			#while True:
				#keep generating random points until one lands inside the outline
				#newpoints[i,0] = minx + random()*(maxx-minx)
				#newpoints[i,1] = miny + random()*(maxy-miny)
				
			if is_point_inside(outline, newpoint):
				newpoints = np.append(newpoints, [newpoint], axis=0)
	
	return newpoints

def replace_with_kmeans_centroid(outline, i):
	
	global midlines
	
	#print(posture_frames[i])
	
	#holds the centroid of the posture (non-weighted)
	centroid = np.zeros(2)
	
	fix_offset_with_preexisting_centroid = False
	
	if fix_offset_with_preexisting_centroid:
		centroid[0] = sum(outline[:,0])/outline.shape[0]
		centroid[1] = sum(outline[:,1])/outline.shape[0]
		
		outline -= centroid
		outline[:,0] += X[posture_frames[i]]
		outline[:,1] += Y[posture_frames[i]]
	else:
		outline += offsets[i,:]
	
	centroid[0] = sum(outline[:,0])/outline.shape[0]
	centroid[1] = sum(outline[:,1])/outline.shape[0]
	
	#all points on and inside the outline
	newpoints = fill_points(outline)
	
	#line with centroids of 2 clusters
	line2, _ = vq.kmeans(newpoints, 2)
	
	toprint = False
	
	if toprint:
		print("Frame#"+str(posture_frames[i]))
		plt.figure(1)
		plt.plot(outline[:,0], outline[:,1], label="Fish outline")
		plt.scatter(X[posture_frames[i]], Y[posture_frames[i]], label="Weighted centroid")
		plt.scatter(centroid[0], centroid[1], label="Outline centroid")
		plt.legend(loc="upper right")
		
		plt.figure(2)
		plt.scatter(newpoints[:,0], newpoints[:,1], label="Interior points")
		plt.scatter(line2[:,0], line2[:,1], label="New centroids")
		plt.plot(line2[:,0], line2[:,1])
		plt.legend(loc="upper right")
		plt.show()
	
	unmerge(line2, posture_frames[i])

def unmerge(centroids, frame):
	
	"""Take the two centroids you got from k-means clustering and try to reassign """
	
	global fish_count, Xall, Yall
	
	inactive_indices = find_inactive_fish(Xall[:,frame])
	prev_frame_nearest = get_nearest(Xall[:,frame-1], Yall[:,frame-1], X[frame], Y[frame])
	
	#default behavior is to not reactivate. this stops edge cases from popping up
	to_reactivate = False
	
	if prev_frame_nearest in inactive_indices:
		#ie the fish that was nearest in the prev frame is now inactive
		to_reactivate = True
		print("Reactivating fish#" + str(prev_frame_nearest) + " at frame " + str(frame))
	
	#this will hold the x,y coords of the fish in the frame before
	#it merged with another fish.
	#it could be such that in the prev frame, the fish was inactive.
	#in that case, we use the current x,y coords instead
	prev_point = []
	
	if X[frame-2] == np.inf:
		if X[frame-1] == np.inf:
			#if no previous point available, compare new centroids with current position
			prev_point.append(X[frame])
			prev_point.append(Y[frame])
		else:
			#if one previous point available, compare new centroids with prev frame position
			prev_point.append(X[frame-1])
			prev_point.append(Y[frame-1])
	else:
		#if two previous points available, compare new centroids with extrapolated point
		prev_point.append(2*X[frame-1] - X[frame-2])
		prev_point.append(2*Y[frame-1] - Y[frame-2])
	
	if get_distance(centroids[0,0], centroids[0,1], prev_point[0], prev_point[1]) < \
	get_distance(centroids[1,0], centroids[1,1], prev_point[0], prev_point[1]):
		Xall[fish_count, frame] = centroids[0,0]
		Yall[fish_count, frame] = centroids[0,1]
		if to_reactivate:
			Xall[prev_frame_nearest, frame] = centroids[1,0]
			Yall[prev_frame_nearest, frame] = centroids[1,1]
	else:
		Xall[fish_count, frame] = centroids[1,0]
		Yall[fish_count, frame] = centroids[1,1]
		if to_reactivate:
			Xall[prev_frame_nearest, frame] = centroids[0,0]
			Yall[prev_frame_nearest, frame] = centroids[0,1]
	
def fill_singleframe_missing():
	for fish in range(Xall.shape[0]):
		for frame in range(1,Xall.shape[1]-1):
			if Xall[fish, frame-1] != np.inf and Xall[fish, frame] == np.inf and Xall[fish, frame+1] != np.inf:
				Xall[fish, frame] = (Xall[fish, frame-1] + Xall[fish, frame+1])/2
				Yall[fish, frame] = (Yall[fish, frame-1] + Yall[fish, frame+1])/2

def reject_frames():
	for frame in range(Xall.shape[1]):
		inactive_fish_count = 0
		for fish in range(Xall.shape[0]):
			if Xall[fish, frame] == np.inf or Xall[fish, frame] == 0:
				inactive_fish_count += 1
		
		if inactive_fish_count > 4:
			Xall[:, frame] = np.nan
			Yall[:, frame] = np.nan

def perim_threshold(do_what, threshold):
	
	"""Calculates the perimeter of each posture and uses a user-supplied threshold
	to find out and analyze bodies comprising of two or more merged fish.
	Once thresholded, the "do_what" definition is called to analyze the outline
	(with the appropriate parameters passed to it)"""
	
	global outline_lengths, outlines, X, Y
	
	outline_iterator = 0
	perims = np.zeros(outline_lengths.shape)
	
	for i in range(len(outline_lengths)):
		
		outline_iterator_next = outline_iterator + int(outline_lengths[i])
		
		outline_slice = outlines[outline_iterator:outline_iterator_next, :]
		
		perims[i] = get_perimeter(outline_slice)
		
		if perims[i] >= threshold:
			do_what(outline_slice, i)
		
		outline_iterator = outline_iterator_next
	
	if False:
		plt.figure(1)
		plt.plot(posture_frames, perims)
		plt.title("Fish posture perimeters")
		plt.xlabel("Frame#")
		plt.ylabel("Perimeter (cm)")
	#plt.show()
	
	return perims
	
def make_periodic(outline):
	outline_new = np.zeros((outline.shape[0]+2,outline.shape[1]))
	outline_new[-1,:] = outline[0,:]
	outline_new[0,:] = outline[-1,:]
	outline_new[1:-1,:] = outline
	
	return outline_new
	
if __name__ == "__main__":

	#plt.style.use('dark_background')
	
	fish_count = int(sys.argv[1])
	
	#set this as the path to the directory holding the .npz files
	abs_path = "/home/shreesh/Videos/data/"
	
	video_filename = str(fish_count)+"_fish.MOV"
	
	print(sys.argv[1])
	
	perim_threshold_value = 0.65 if fish_count == 15 else 0.92 if fish_count == 30 else 1
	
	print(perim_threshold)
	
	perims = np.array([])
	
	fish_filename = abs_path + video_filename + "_fish0.npz"
		
	load_file(fish_filename)
	
	max_frames = 4000 if fish_count == 30 else 6000
	
	Xall = np.zeros((fish_count, max_frames))
	Yall = np.zeros((fish_count, max_frames))
	
	print("Xall shape: ", Xall.shape)
	
	Xall[0,0:len(X)] = X
	Yall[0,0:len(Y)] = Y
	
	for count in range(1,fish_count):
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		load_file(fish_filename)
		
		print("X shape: ", X.shape)
		
		Xall[count,0:len(X)] = X
		Yall[count,0:len(Y)] = Y
	
	for count in range(fish_count):
		
		print("Processing fish#"+str(count))
		
		fish_filename = abs_path + video_filename + "_posture_fish" + str(count) + ".npz"
		
		load_posture_file(fish_filename)
		
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		load_file(fish_filename)
		
		perim_threshold(replace_with_kmeans_centroid, perim_threshold_value)
		
		#perims = np.append(perims, perim_threshold(replace_with_kmeans_centroid, 50*cm_per_pixel))
		
		if False:
			plt.figure(0)
			plt.plot(X)
			plt.title("X")
			#plt.legend(loc="upper right")
			plt.figure(1)
			plt.plot(Y)
			plt.title("Y")
			#plt.legend(loc="upper right")
			plt.figure(2)
			plt.plot(X,Y)
			plt.figure(3)
			plt.plot(X-X_copy)
			plt.title("X diff")
			plt.figure(4)
			plt.plot(Y-Y_copy)
			plt.title("Y diff")
	
	fill_singleframe_missing()
	reject_frames()
	
	if True:
		plt.figure(0)
		for i in range(fish_count):
			plt.plot(Xall[i,:])
		plt.title("X")
		plt.figure(1)
		for i in range(fish_count):
			plt.plot(Yall[i,:])
		plt.title("Y")
	
	#inf_cnt = np.sum(np.isnan(Xall))
	inf_cnt = np.sum(np.isinf(Xall))
	
	print("Inf cnt: " + str(inf_cnt))
	
	print("Saving file: posAll_"+str(fish_count)+".npz")
	
	np.savez("posAll_"+str(fish_count)+".npz", Xall=Xall, Yall=Yall)
	
	#plt.figure(1)
	#plt.plot(perims)
	#plt.title("Fish posture perimeters")
	#plt.xlabel("Frame#")
	#plt.ylabel("Perimeter (cm)")
	
	#plt.figure(2)
	#plt.hist(perims, bins=40)
	#plt.title("Perimeter frequencies")
	#plt.xlabel("Perimeter (cm)")
	#plt.ylabel("Frequency")
	#plt.show()
	
	print("Avg perimeter: " + str(np.mean(perims)))
	print("Std dev      : " + str(np.std(perims)))
	
	plt.show()
