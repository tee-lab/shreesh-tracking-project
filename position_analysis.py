import numpy as np
import matplotlib.pyplot as plt

def load_file(fish_filename):
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]
	
	#get the vector representing the frames of the video
	frame_vec = npz["frame"]
	
	return X, Y, frame_vec

def find_skips(X, Y, threshold, found_indices):
	
	#this array holds index positions where ID skips have taken place
	skips = np.zeros(X.shape[0])
	
	skip_count = 0
	
	#scan X and Y vectors for skips
	
	for i in range(0, X.shape[0]-1):
		#first make sure finite values are being dealt with
		if X[i] != np.inf and X[i+1] != np.inf:
			#if motion in one frame is abnormally large, register as ID skip
			if abs(X[i+1] - X[i]) > threshold and skips[i] != 1:
				skips[i] = 1
				skip_count += 1
				found_indices.append(i)
	
	for i in range(0, Y.shape[0]-1):
		if Y[i] != np.inf and Y[i+1] != np.inf:
			if abs(Y[i+1] - Y[i]) > threshold and skips[i] != 1:
				skips[i] = 1
				skip_count += 1
				found_indices.append(i)
	
	return skips, skip_count, found_indices

def find_skips_radial(X, Y, threshold, found_indices):
	
	#this array holds index positions where ID skips have taken place
	skips = np.zeros(X.shape[0])
	
	skip_count = 0
	
	#scan X and Y vectors for skips
	
	for i in range(0, X.shape[0]-1):
		#first make sure finite values are being dealt with
		if X[i] != np.inf and X[i+1] != np.inf and Y[i] != np.inf and Y[i+1] != np.inf:
			#if motion in one frame is abnormally large, register as ID skip
			if np.sqrt((X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2) > threshold and skips[i] != 1:
				skips[i] = 1
				skip_count += 1
				found_indices.append(i)
	
	return skips, skip_count, found_indices

def get_skip_count(skips, found_indices):
	count = 0
	for i in range(len(skips)):
		if i not in found_indices and skips[i] == 1:
			count += 1
			found_indices.append(i)
	
	return count, found_indices

def fill_in_gaps(X):
	
	#bool value to remember if currently inside a gap	
	inside_a_gap = False
	#ideally should be 0, but have to consider the edge case
	#of a body starting out in the inactive state
	first_active_index = -1
	
	#iterate up to the first finite value (ie when fish is first active)
	for i in range(len(X)):
		if X[i] != np.inf:
			first_active_index = i
			break
	
	#this value is used to remember the last index
	#where the fish was active
	last_active_index = first_active_index
	
	for i in range(first_active_index, len(X)):
		
		if inside_a_gap:
			
			#if gap is complete, fill it in using linear interpolation
			if X[i] != np.inf:
				
				#increment per frame
				increment = (X[i] - X[last_active_index])/(i - last_active_index)
				
				#now perform linear interpolation to fill in the gaps
				for j in range(last_active_index+1, i):
					X[j] = X[last_active_index] + (j-last_active_index)*increment
				
				inside_a_gap = False
			
			else:
				continue
		
		#if non-finite value encountered, you have entered a gap
		elif X[i] == np.inf:
			inside_a_gap = True
			last_active_index = i-1
		else:
			continue
	
	return X

def moving_average(X, span):
	return np.convolve(X, np.ones(span), 'valid') / span			

def main():
	
	#set this as the path to the directory holding the .npz files
	abs_path = "/home/shreesh/Videos/data/"
	video_filename = "30_fish.MOV"

	max_fish_count = 30

	#this is the max possible legitimate displacement in one frame
	#any motion greater than this value is marked as an ID skip
	skip_threshold = 0.5

	total_skip_count = 0
	skip_positions = []
	
	for count in range(0, max_fish_count):
		
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		X, Y, frame_vec = load_file(fish_filename)
		
		#X = fill_in_gaps(X)
		#Y = fill_in_gaps(Y)
		
		skips, current_skip_count, skip_positions = find_skips_radial(X, Y, skip_threshold, skip_positions)
		
		#current_skip_count, skip_positions = get_skip_count(skips, skip_positions)
		
		total_skip_count += current_skip_count
		
		#plot X position data
		plt.figure(1)
		plt.plot(frame_vec, X)
		#plt.plot(frame_vec[skips.astype(bool)], X[skips.astype(bool)])
		plt.plot(frame_vec, skips*20)
		plt.title("X position and skips vs Frames")
		plt.xlabel("Frame #")
		plt.ylabel("Xpos")
		
		#plot Y position data
		plt.figure(2)
		plt.plot(frame_vec, Y)
		plt.plot(frame_vec, skips*20)
		#plt.plot(frame_vec[skips.astype(bool)], Y[skips.astype(bool)])
		plt.title("Y position and skips vs Frames")
		plt.xlabel("Frame #")
		plt.ylabel("Ypos")

	print("Total skip count is: " + str(total_skip_count))
	print("Skips happened at frames:")
	skip_positions = list(set(skip_positions))
	skip_positions.sort()
	print(skip_positions)
	
	plt.show()

if __name__ == "__main__":
    main()
	