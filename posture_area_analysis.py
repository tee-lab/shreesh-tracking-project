import numpy as np
import matplotlib.pyplot as plt

def load_posture_file(fish_filename):
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	areas = npz["posture_area"]
	
	#get the vector representing the frames of the video
	frame_vec = npz["frames"]
	
	return areas, frame_vec

def load_position_file(fish_filename):
	#load the npz file of the given fish
	npz = np.load(fish_filename)
	
	#extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]
	
	#get the vector representing the frames of the video
	frame_vec = npz["frame"]
	
	return X, Y, frame_vec

def find_inactive_fish(fish_positions):
	inactive_indices = []
	
	for i in range(len(fish_positions)):
		if fish_positions[i] == np.inf:
			inactive_indices.append(i)
	
	return inactive_indices

#find nearest neighbor of fish with fish index: fish_index at position Xpos,Ypos
def find_nearest_neighbor(fish_index, fish_pos, all_pos):
	
	min_dist = 100000
	nearest_index = -1
	
	for i in range(all_pos.shape[0]):
		if i != fish_index:
			if all_pos[i][0] != np.inf and all_pos[i][1] != np.inf:
				dist = (all_pos[i][0] - fish_pos[0])**2 + (all_pos[i][1] - fish_pos[1])**2
				if min_dist > dist:
					min_dist = dist
					nearest_index = i
	
	return nearest_index

def main(plot):
	
	plt.style.use('dark_background')
	
	#set this as the path to the directory holding the .npz files
	abs_path = "./data/"
	video_filename = "30_fish.MOV"

	max_fish_count = 1
	
	max_frames = 3137
	
	all_areas = np.zeros((max_fish_count,max_frames))
	
	average_areas = np.zeros(max_frames)
	
	all_pos = np.zeros((max_fish_count,max_frames, 2))
	
	for count in range(0, max_fish_count):
		
		fish_filename = abs_path + video_filename + "_posture_fish" + str(count) + ".npz"
		
		areas, frames1 = load_posture_file(fish_filename)
		
		max_index = max_frames if max_frames < len(areas) else len(areas)
		
		all_areas[count,0:max_index] = areas[0:max_index]
		
		fish_filename = abs_path + video_filename + "_fish" + str(count) + ".npz"
		
		X, Y, frames2 = load_position_file(fish_filename)
		
		all_pos[count, 0:max_index, 0] = X[0:max_index]
		all_pos[count, 0:max_index, 1] = Y[0:max_index]
		
		if plot:
			plt.figure(1)
			plt.plot(all_areas[count,:])
			plt.title("Posture Areas vs Frame#")
			plt.xlabel("Frame #")
			plt.ylabel("Posture area (cm^2)")
	
	#set after looking at the plot
	max_posture_area = 100
	
	#print(all_pos[1,1,:])
	#print(all_pos[:,1,:])
	
	merge_count = 0
	
	merges = np.zeros(max_frames)
	
	for i in range(max_frames-1):
		average_area = 0
		average_count = 0
		for count in range(max_fish_count):
			
			#find fish who has roughly doubled in size in the next frame
			if all_areas[count, i+1] != np.inf and all_areas[count, i] != np.inf \
			and all_areas[count, i+1] > 0 and all_areas[count, i] > 0 \
			and all_areas[count,i+1] >= 2*all_areas[count, i] and all_areas[count,i+1] > max_posture_area:
				
				#find fish that went inactive in the next frame
				inactive_indices = find_inactive_fish(all_pos[:, i+1, 0])
				
				#find nearest neighbor among the fish that went inactive
				nearest_fish_index = find_nearest_neighbor(count, all_pos[count, i, :], all_pos[inactive_indices, i, :])
				
				if nearest_fish_index > -1:
					print("Frame, Fish: " + str(i) + " " + str(count))
					print("    Nearest: " + str(nearest_fish_index))
					
					#if all_pos[nearest_fish_index, i+1, 0] == np.inf:
					merge_count += 1
				
					if merges[i] != 1:
						merges[i] = 1
		
		if average_count > 0:
			average_areas[i] = average_area/average_count
		
	print(merge_count)
	
	print("Final average area: " + str(sum(average_areas)/len(average_areas)))
	
	plt.figure(1)
	plt.plot(merges)
	#plt.title("Merges")
	
	#plt.figure(2)
	#plt.plot(average_areas)
	#plt.title("Average areas")
	
	plt.show()
	
	
	#for count in range(max_fish_count):
		

if __name__ == "__main__":
    main(True)