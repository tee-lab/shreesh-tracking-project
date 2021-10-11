import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq

outlines = np.array(1)
midlines = np.array(1)
outline_lengths = np.array(1)
midline_lengths = np.array(1)
outline_iterator = 0
midline_iterator = 0
X = np.array(1)
Y = np.array(1)

offsets = np.array(1)

#the following two defs are utility functions

def moving_average(X, span):
	return np.convolve(X, np.ones(span), 'valid') / span

def get_distance(x1, y1, x2, y2):
	return ( (x2-x1)**2 + (y2-y1)**2 )**0.5

def load_posture_file(fish_filename):
	
	global outlines, outline_lengths
	global midlines, midline_lengths
	global offsets
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	outlines = npz["outline_points"]
	midlines = npz["midline_points"]
	
	outline_lengths = npz["outline_lengths"]
	midline_lengths = npz["midline_lengths"]
	
	offsets = npz["offset"]

def load_file(fish_filename):

	global X, Y
	
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X"]#wcentroid"]
	Y = npz["Y"]#wcentroid"]
	#scale up to pixel values using the cm_per_pixel parameter from Trex
	X /= 0.018428
	Y /= 0.018428

def get_perimeter(curr_outline):
	
	curr_outline_new = np.zeros((curr_outline.shape[0]+1,2))
	curr_outline_new[-1,:] = curr_outline[0,:]
	curr_outline_new[0:-1,:] = curr_outline
	
	segment_lengths = get_distance(curr_outline_new[0:-1,0], curr_outline_new[0:-1,1], curr_outline_new[1:,0], curr_outline_new[1:,1])
	
	return sum(segment_lengths)

def kmeans_clustering(outline, count=2, whiten=False):
	
	if whiten:
		outline = vq.whiten(outline)
	
	line, distort = vq.kmeans(outline, count)
	
	return line

def replace_with_kmeans_centroid(outline, i):
	
	outline = offsets[i,:] + outline
	
	line = kmeans_clustering(outline)
	
	if get_distance(line[0,0], line[0,1], X[i-1], Y[i-1]) > get_distance(line[1,0], line[1,1], X[i-1], Y[i-1]):
		X[i] = line[0,0]
		Y[i] = line[0,1]
	else:
		X[i] = line[1,0]
		Y[i] = line[1,1]

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
	
def make_periodic(outline):
	outline_new = np.zeros((outline.shape[0]+2,outline.shape[1]))
	outline_new[-1,:] = outline[0,:]
	outline_new[0,:] = outline[-1,:]
	outline_new[1:-1,:] = outline
	
	return outline_new

def plot_curvature(outline, i):
	
	#first make the array continuous/periodic
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
	
	for count in range(0, max_fish_count):
		
		fish_filename = abs_path + video_filename + "_posture_fish" + str(count) + ".npz"
		
		load_posture_file(fish_filename)
		
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		load_file(fish_filename)
		
		perim_threshold(replace_with_kmeans_centroid, 50)
		
		plt.figure(0)
		plt.plot(X)
		plt.title("X")
		plt.figure(1)
		plt.plot(Y)
		plt.title("Y")
		#plt.figure(2)
		#plt.plot(offsets[:,0])
		#plt.title("Offset X")
		#plt.figure(3)
		#plt.plot(offsets[:,1])
		#plt.title("Offset Y")
	
	plt.show()