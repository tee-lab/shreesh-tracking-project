import numpy as np
from random import random
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
from scipy.cluster.hierarchy import fclusterdata

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
pos_frames_all = []

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

	global X, Y, X_copy, Y_copy
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]
	
	X_copy = npz["X#wcentroid"]
	Y_copy = npz["Y#wcentroid"]

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
	
	#cluster_indices = fclusterdata(newpoints, t=2, criterion='maxclust')#, metric='euclidean')
	#cluster_indices -= 1
	#print(cluster_indices)
	
	#line2 = np.zeros((2,2))
	
	#for clust in range(2):
	#	cluster = newpoints[(cluster_indices==clust).astype(bool),:]
	#	line2[clust,:] = np.sum(cluster, axis=0)/cluster.shape[0]
	
	find_4means_centroids = False
	
	if find_4means_centroids:
		
		#here we further subdivide the body into 4 parts
		#to hopefully find the two heads and two tails
		
		#line with centroid of 4 clusters
		line4, reject = vq.kmeans(newpoints, 4)
		
		#holds all pairs of points resulting from 2 clusters 
		#and pairs of points from 4 clusters
		final_lines = np.zeros((4,2,2))
		
		final_lines[0,0,:] = (line4[0,:] + line4[1,:])/2
		final_lines[0,1,:] = (line4[2,:] + line4[3,:])/2
		
		final_lines[1,0,:] = (line4[0,:] + line4[2,:])/2
		final_lines[1,1,:] = (line4[1,:] + line4[3,:])/2
		
		final_lines[2,0,:] = (line4[0,:] + line4[3,:])/2
		final_lines[2,1,:] = (line4[2,:] + line4[1,:])/2
		
		final_lines[3,:,:] = line2
		
		#to find which of these pairs of points form a line that
		#is most perpendicular to the first line of the midline
		head_slope = get_slope(midlines[i,0:2,:])
		head_slope_perp = head_slope
		
		line_slopes = np.zeros(4)
		final_line_lengths = np.zeros(4)
		
		for j in range(4):
			line_slopes[j] = get_slope(final_lines[j,:,:])
			final_line_lengths[j] = get_distance_nparray(final_lines[j,:,:])
		
		ang_diffs = np.arctan( np.abs( (head_slope_perp - line_slopes)/(1 + head_slope_perp*line_slopes) ) )
		
		best_line_index = np.argsort(ang_diffs)[0]
		shortest_line_index = np.argsort(final_line_lengths)[0]
		
		if best_line_index == shortest_line_index:
			best_line_index = np.argsort(ang_diffs)[2]
	else:
		final_lines = np.zeros((1,2,2))
		final_lines[0,:,:] = line2
		best_line_index = 0
	
	toprint = True
	
	if toprint:
		print("Frame#"+str(posture_frames[i]))
		plt.figure(1)
		plt.plot(outline[:,0], outline[:,1], label="Fish outline")
		plt.scatter(X[posture_frames[i]], Y[posture_frames[i]], label="Weighted centroid")
		plt.scatter(centroid[0], centroid[1], label="Outline centroid")
		plt.legend(loc="upper right")
		
		plt.figure(2)
		plt.scatter(newpoints[:,0], newpoints[:,1], label="Interior points")
		plt.scatter(final_lines[best_line_index,:,0], final_lines[best_line_index,:,1], label="New centroids")
		plt.plot(final_lines[best_line_index,:,0], final_lines[best_line_index,:,1])
		plt.legend(loc="upper right")
		plt.show()
	
	replace_points(line2, posture_frames[i])
	
	#find_inactive_fish(Xall[:][
	
	#if X[posture_frames[i]-1] == np.inf or Y[posture_frames[i]-1] == np.inf:
	#	prev_point.append(X[posture_frames[i]])
	#	prev_point.append(Y[posture_frames[i]])
	#else:
	#	prev_point.append(X[posture_frames[i]-1])
	#	prev_point.append(Y[posture_frames[i]-1])
	
	#if get_distance(final_lines[best_line_index,0,0], final_lines[best_line_index,0,1], prev_point[0], prev_point[1]) > \
	#get_distance(final_lines[best_line_index, 1,0], final_lines[best_line_index,1,1], prev_point[0], prev_point[1]):
	#	X[posture_frames[i]] = final_lines[best_line_index,0,0]
	#	Y[posture_frames[i]] = final_lines[best_line_index,0,1]
	#else:
	#	X[posture_frames[i]] = final_lines[best_line_index,1,0]
	#	Y[posture_frames[i]] = final_lines[best_line_index,1,1]

def replace_points(line, i):
	
	global fish_count, Xall, Yall
	
	inactive_indices = find_inactive_fish(Xall[:,i])
	prev_frame_nearest = get_nearest(Xall[:,i-1], Yall[:,i-1], X[i], Y[i])
	
	to_reactivate = False
	
	if prev_frame_nearest in inactive_indices:
		#ie the fish that was nearest in the prev frame is now
		to_reactivate = True
		print("Reactivated fish#" + str(prev_frame_nearest) + " at frame " + str(i))
	
	#this will hold the x,y coords of the fish in the frame before
	#it merged with another fish.
	#it could be such that in the prev frame, the fish was inactive.
	#in that case, we use the current x,y coords instead
	prev_point = []

	if X[i-1] == np.inf or Y[i-1] == np.inf:
		prev_point.append(X[i])
		prev_point.append(Y[i])
	else:
		prev_point.append(X[i-1])
		prev_point.append(Y[i-1])
	
	if get_distance(line[0,0], line[0,1], prev_point[0], prev_point[1]) < \
	get_distance(line[1,0], line[1,1], prev_point[0], prev_point[1]):
		Xall[fish_count, i] = line[0,0]
		Yall[fish_count, i] = line[0,1]
		if to_reactivate:
			Xall[prev_frame_nearest, i] = line[1,0]
			Yall[prev_frame_nearest, i] = line[1,1]
	else:
		Xall[fish_count, i] = line[1,0]
		Yall[fish_count, i] = line[1,1]
		if to_reactivate:
			Xall[prev_frame_nearest, i] = line[0,0]
			Yall[prev_frame_nearest, i] = line[0,1]

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

def plot_curvature(outline, i):
	
	#first make the array periodic
	outline_new = make_periodic(outline)
	outline_new[1:-1,0] = moving_average(outline_new[:,0], 3)
	outline_new[1:-1,1] = moving_average(outline_new[:,1], 3)
	outline_new = make_periodic(outline_new[1:-1,:])
	
	#these hold the first and second derivatives of the x and y components of the outline
	der = np.zeros(outline.shape)
	dder = np.zeros(outline.shape)
	
	#forward difference
	der = (outline_new[2:,:] - outline_new[0:-2,:])/2
	der_new = make_periodic(der)
	
	dder = (der_new[2:,:] - der_new[0:-2,:])/2
	
	#find the curvature of the outline along its length
	curvature = (der[:,0]*dder[:,1] - der[:,1]*dder[:,0])/((der[:,0]**2 + der[:,1]**2)**1.5)
	periodic_curvature = np.zeros(len(curvature)+2)
	periodic_curvature[0] = curvature[-1]
	periodic_curvature[-1] = curvature[0]
	periodic_curvature[1:-1] = curvature
	
	#find the derivative of the curvature
	curve_der = periodic_curvature[1:-1] - periodic_curvature[0:-2]
	
	plt.figure(1)
	plt.plot(curve_der)
	plt.figure(2)
	plt.plot(outline[:,0], outline[:,1])
	plt.show()
	
if __name__ == "__main__":

	#plt.style.use('dark_background')
	
    #set this as the path to the directory holding the .npz files
	abs_path = "/home/shreesh/Videos/data/"
	video_filename = "30_fish.MOV"

	max_fish_count = 30
	
	print("Perim threshold = " + str(50*cm_per_pixel))
	
	perims = np.array([])
	
	for count in range(max_fish_count):
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		load_file(fish_filename)
		
		if count == 0:
			Xall = np.array([X])
			Yall = np.array([Y])
		else:
			Xall = np.append(Xall, [X[0:Xall.shape[1]]], axis=0)
			Yall = np.append(Yall, [Y[0:Xall.shape[1]]], axis=0)
	
	for count in range(max_fish_count):
		
		fish_count = count
		
		print("Processing fish#"+str(count))
		
		fish_filename = abs_path + video_filename + "_posture_fish" + str(count) + ".npz"
		
		load_posture_file(fish_filename)
		
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		load_file(fish_filename)
		
		perim_threshold(replace_with_kmeans_centroid, 50*cm_per_pixel)
		
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
	
	if True:
		plt.figure(0)
		for i in range(max_fish_count):
			plt.plot(Xall[i,:])
		plt.title("X")
		plt.figure(1)
		for i in range(max_fish_count):
			plt.plot(Yall[i,:])
		plt.title("Y")
	
	np.savez("posAll.npz", Xall=Xall, Yall=Yall)
	
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
	
	#print("Avg perimeter: " + str(np.mean(perims)))
	#print("Std dev      : " + str(np.std(perims)))
	
	plt.show()