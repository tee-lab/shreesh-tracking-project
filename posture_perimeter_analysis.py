import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pltPath
import scipy.cluster.vq as vq
import utility as utils
#from scipy.cluster.hierarchy import fclusterdata

switch_array = []

posture_frames = np.array(1)
outlines = np.array(1)

outline_lengths = 0

outline_iterator = 0

X = np.array(1)
Y = np.array(1)

current_fish_index = -1

Xall = np.array([])
Yall = np.array([])
frames_array = []

offsets = np.array(1)

def get_nearest(fish_listX, fish_listY, x, y):
	dists = utils.get_distance(fish_listX, fish_listY, x, y)
	return np.argsort(dists)

def find_inactive_fish(fish_list):
	finite_indices = np.isfinite(fish_list)
	return np.where(finite_indices == False)[0]

def fill_points(outline):

	"""This function takes in an array of points representing a continuous
	and periodic outline of a body and outputs a list of points inside it
	spaced in a grid-like fashion"""

	# find the the bounding box of the outline
	minx = min(outline[:,0])
	miny = min(outline[:,1])
	maxx = max(outline[:,0])
	maxy = max(outline[:,1])

	# the number of points in the x and y axis of the bounding box
	numpoints = int(2.5*outline.shape[0]**0.5)

	xstep = (maxx-minx)/numpoints
	ystep = (maxy-miny)/numpoints

	# create a list of points on the grid inside the bounding box
	gridpoints = np.mgrid[minx:maxx:xstep,miny:maxy:ystep]
	gridpoints = gridpoints.reshape((2,gridpoints.shape[1]*gridpoints.shape[2])).T

	# find the points lying inside the outline
	path = pltPath.Path(outline)
	inside = path.contains_points(gridpoints)
	inside_points = gridpoints[inside,:]

	return inside_points

def plot_posture_points(outline, points, centroids):
	plt.figure(1)
	plt.scatter(outline[:,0], outline[:,1])
	plt.scatter(points[:,0], points[:,1])
	if centroids != None:
		plt.scatter(centroids[:,0], centroids[:,1])
	plt.show()

def replace_with_kmeans_centroid(outline, i):

	outline += offsets[i,:]

	#all points inside the outline
	newpoints = fill_points(outline)

	#line with centroids of 2 clusters
	line2, _ = vq.kmeans(newpoints, 2)

	# plot_posture_points(outline, newpoints, line2)

	unmerge(line2, posture_frames[i])

def unmerge(centroids, frame):

	"""Take the two centroids you got from k-means clustering and try to reassign """

	global current_fish_index, Xall, Yall

	inactive_indices = find_inactive_fish(Xall[:,frame])
	prev_frame_nearest_fish = get_nearest(Xall[:,frame-1], Yall[:,frame-1],\
		Xall[current_fish_index, frame], Yall[current_fish_index, frame])

	prev_frame_nearest = prev_frame_nearest_fish[1]

	# default behavior is to not reactivate. this stops edge cases from popping up
	to_reactivate = False

	for i in range(1,4):
		if i >= len(prev_frame_nearest_fish):
			break
		prev_frame_nearest = prev_frame_nearest_fish[i]
		if prev_frame_nearest in inactive_indices:
			# ie the fish that was nearest in the prev frame is now inactive
			to_reactivate = True
			print("Reactivating fish#" + str(prev_frame_nearest) + " at frame " + str(frame))

	# this will hold the x,y coords of the fish in the frame before
	# it merged with another fish.
	# it could be such that in the prev frame, the fish was inactive.
	# in that case, we use the current x,y coords instead
	prev_point = []

	if Xall[current_fish_index, frame-1] == np.inf:
			# if previous point not available, compare new centroids with current position
			prev_point.append(Xall[current_fish_index, frame])
			prev_point.append(Yall[current_fish_index, frame])
	elif Xall[current_fish_index, frame-2] == np.inf:
			# if only one previous point available, compare new centroids with prev frame position
			prev_point.append(Xall[current_fish_index, frame-1])
			prev_point.append(Yall[current_fish_index, frame-1])
	else:
		# if two previous points available, compare new centroids with extrapolated point of previous two frames
		prev_point.append(2*Xall[current_fish_index, frame-1] - Xall[current_fish_index, frame-2])
		prev_point.append(2*Yall[current_fish_index, frame-1] - Yall[current_fish_index, frame-2])

	if utils.get_distance(centroids[0,0], centroids[0,1], prev_point[0], prev_point[1]) < \
	utils.get_distance(centroids[1,0], centroids[1,1], prev_point[0], prev_point[1]):
		Xall[current_fish_index, frame] = centroids[0,0]
		Yall[current_fish_index, frame] = centroids[0,1]
		if to_reactivate:
			Xall[prev_frame_nearest, frame] = centroids[1,0]
			Yall[prev_frame_nearest, frame] = centroids[1,1]
	else:
		Xall[current_fish_index, frame] = centroids[1,0]
		Yall[current_fish_index, frame] = centroids[1,1]
		if to_reactivate:
			Xall[prev_frame_nearest, frame] = centroids[0,0]
			Yall[prev_frame_nearest, frame] = centroids[0,1]

	switch = utils.IdSwitch()
	switch.set_type(2)
	switch.set_frame(frame)
	switch.set_fish(current_fish_index, prev_frame_nearest)

	switch_array.append(switch)

def fix_id_switches(switch_array):

	switch_array.sort(key = lambda x: (x.frame_num, x.fish1))

	removed = []

	for switch in switch_array:

		fish1 = switch.fish1
		fish2 = switch.fish2
		frame = switch.frame_num

		if Xall[fish1, frame-1] == np.inf and Xall[fish1, frame-1] == np.inf:
			switch_array.remove(switch)
			print('Removed:')
			switch.display()
			removed.append(switch)
			continue # cannot fix if an ID switch occured in this scenario

		if frame+1 == Xall.shape[1]:
			switch_array.remove(switch)
			print('Removed:')
			switch.display()
			removed.append(switch)
			continue

		extrapolated_next_point = [\
			2*Xall[fish1, frame] - Xall[fish1, frame-1],\
			2*Yall[fish2, frame] - Yall[fish2, frame-1] \
			]

		if Xall[fish1, frame+1] == np.inf:
			# if the original body is inactive in the next frame,
			# assign the extrapolated position to it
			Xall[fish1, frame+1] = extrapolated_next_point[0]
			Yall[fish1, frame+1] = extrapolated_next_point[1]
		else:
			# first check if unmerged fish is active in the next frame
			if Xall[fish2, frame+1] != np.inf:
				# then check if ID switch has occured by using the extrapolated point
				if utils.get_distance(extrapolated_next_point[0], extrapolated_next_point[1],\
					Xall[fish1, frame+1], Yall[fish1, frame+1]) > \
					utils.get_distance(extrapolated_next_point[0], extrapolated_next_point[1],\
					Xall[fish2, frame+1], Yall[fish2, frame+1]):
					continue

		switch_array.remove(switch)
		print('Removed:')
		switch.display()
		removed.append(switch)

	return switch_array, removed


def switch(index1, index2, framenum):

	"""Switch the trajectories of fish1 and fish2 starting from a given frame"""

	temp = Xall[index1, framenum:Xall.shape[1]]
	Xall[index1, framenum:Xall.shape[1]] = Xall[index2, framenum:Xall.shape[1]]
	Xall[index2, framenum:Xall.shape[1]] = temp

def fill_singleframe_missing():
	for fish in range(Xall.shape[0]):
		for frame in range(1,Xall.shape[1]-1):
			if Xall[fish, frame-1] != np.inf and Xall[fish, frame] == np.inf and Xall[fish, frame+1] != np.inf:
				Xall[fish, frame] = (Xall[fish, frame-1] + Xall[fish, frame+1])/2
				Yall[fish, frame] = (Yall[fish, frame-1] + Yall[fish, frame+1])/2

# def reject_frames():
# 	for frame in range(Xall.shape[1]):
# 		inactive_fish_count = 0
# 		for fish in range(Xall.shape[0]):
# 			if Xall[fish, frame] == np.inf or Xall[fish, frame] == 0:
# 				inactive_fish_count += 1
#
# 		if inactive_fish_count > 4:
# 			Xall[:, frame] = np.inf
# 			Yall[:, frame] = np.inf

def perim_threshold(do_what, threshold):

	"""Calculates the perimeter of each posture and uses a user-supplied threshold
	to find out and analyze bodies comprising of two or more merged fish.
	Once thresholded, the "do_what" definition is called to analyze the outline
	(with the appropriate parameters passed to it)"""

	global Xall, outline_lengths, outlines, posture_frames

	outline_iterator = 0
	perims = np.zeros(outline_lengths.shape)

	for i in range(len(outline_lengths)):

		outline_iterator_next = outline_iterator + int(outline_lengths[i])

		if not(posture_frames[i] >= Xall.shape[1] or reject_frames[posture_frames[i]]):

			outline_slice = outlines[outline_iterator:outline_iterator_next, :]

			perims[i] = utils.get_perimeter(outline_slice)

			if perims[i] >= threshold:
				do_what(outline_slice, i)

		outline_iterator = outline_iterator_next

	return perims

if __name__ == "__main__":

	filename = sys.argv[1]

	cfg = utils.Config("config_files/"+filename+".csv")

	max_fish_count = cfg.fish_count
	perim_threshold_value = cfg.perim_thresh

	perims = np.array([])

	Xall, Yall, reject_frames, first_frame = utils.collate(cfg, fill_gaps=False)

	if False:

		plt.figure(0)
		for i in range(cfg.fish_count):
			plt.plot(Xall[i,:])
		plt.title("X")
		plt.figure(1)
		for i in range(cfg.fish_count):
			plt.plot(Yall[i,:])
		plt.title("Y")
		plt.show()

	current_fish_index = 0

	for count in range(cfg.fish_count):

		print("Processing fish#"+str(count))

		outlines, outline_lengths, offsets, posture_frames = utils.load_posture_file(cfg, count)

		X = Xall[count,:]
		Y = Yall[count,:]

		current_fish_index = count

		perim_threshold(replace_with_kmeans_centroid, perim_threshold_value)

		if False:
			plt.figure(0)
			plt.plot(X)
			plt.title("X")
			#plt.legend(loc="upper right")
			plt.figure(1)
			plt.plot(Y)
			plt.title("Y")
			#plt.legend(loc="upper right")

	#fill_singleframe_missing()
	#reject_frames()

	if True:
		plt.figure(0)
		for i in range(cfg.fish_count):
			plt.plot(Xall[i,:])
		plt.title("X")
		plt.figure(1)
		for i in range(cfg.fish_count):
			plt.plot(Yall[i,:])
		plt.title("Y")

	switch_array, removed = fix_id_switches(switch_array)

	for switch in switch_array:
		switch.frame_num += first_frame
		switch.display()

	print("Total: ", len(switch_array))
	print("Removed: ", len(removed))

	#print("Saving file: posAll__"+str(fish_count)+".npz")

	#np.savez("posAll_"+str(fish_count)+".npz", Xall=Xall, Yall=Yall)

	plt.show()

	#for switch in switch_array:
	#	switch.display()

	utils.write_csv(switch_array, "csv_files/"+filename+"/perim_switches_"+filename+".csv")
	utils.write_csv(removed, "csv_files/"+filename+"/removed_perim_switches_"+filename+".csv")


#use 3555-4814 frames
