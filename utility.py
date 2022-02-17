import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

class Config:
	"""A class that defines config entries and whose objects are
	used by the scripts to read said entries"""
	def __init__(self):
		self.fish_count = 0
		self.video_name = ""
		self.video_dir = ""
		self.jump_thresh = 0
		self.perim_thresh = 0

	def __init__(self, filename):
		self.read_csv(filename)

	def set_fish_count(self, count):
		self.fish_count = count

	def set_name(self, name):
		self.video_name = name

	def set_dir(self, dir):
		self.video_dir = dir

	def set_thresholds(self, jump, perim):
		self.jump_thresh = jump
		self.perim_thresh = perim

	def read_csv(self, csv_name):
		with open(csv_name, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			for firstline in csvreader:
				self.fish_count = int(firstline[0])
				self.video_dir = firstline[1]
				self.video_name = firstline[2]
				self.jump_thresh = float(firstline[3])
				self.perim_thresh = float(firstline[4])

class IdSwitch:
	"""A class that represents an ID switch in the data"""
	frame_num = -1
	def __init__(self):
		self.type = -1
		self.frame_num = -1
		self.fish1 = -1
		self.fish2 = -1

	def set_type(self, switch_type):
			self.type = switch_type

	def set_frame(self, frame):
		self.frame_num = frame

	def set_fish(self, fish1, fish2):
		self.fish1 = fish1
		self.fish2 = fish2

	def display(self):
		print("Type: ", self.type)
		print("    at frame: ", self.frame_num)
		print("    between fish", self.fish1, "and fish", self.fish2)

	def get_dict_entry(self):
		return {'Type': str(self.type),\
			'Frame': str(self.frame_num),\
			'Fish1': str(self.fish1),\
			'Fish2': str(self.fish2)}

def get_distance(x1, y1, x2, y2):
	return ( (x2-x1)**2 + (y2-y1)**2 )**0.5

def get_perimeter(outline):

	"""This function takes in an outline (periodic) and outputs its perimeter."""

	#to store periodic version of outline for processing purposes
	outline_new = np.zeros((outline.shape[0]+1,2))
	outline_new[-1,:] = outline[0,:]
	outline_new[0:-1,:] = outline

	segment_lengths = get_distance(outline_new[0:-1,0], outline_new[0:-1,1], outline_new[1:,0], outline_new[1:,1])

	return sum(segment_lengths)

def write_csv(switch_array, filename):

	dict_list = []
	for switch in switch_array:
		dict_list.append(switch.get_dict_entry())

	fields = ['Type', 'Frame', 'Fish1', 'Fish2']

	with open(filename, 'w') as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(dict_list)

def read_csv(filename):

	switch_array = []

	with open(filename, 'r') as csv_file:
		csv_file_ = csv.DictReader(csv_file)
		for dict_entry in csv_file_:
			switch = IdSwitch()
			switch.set_type(int(dict_entry['Type']))
			switch.set_frame(int(dict_entry['Frame']))
			switch.set_fish(int(dict_entry['Fish1']), int(dict_entry['Fish2']))
			switch_array.append(switch)
			#switch.display()

		#print(switch_array)

	return switch_array

def load_posture_file(config_file, count):

	cm_per_pixel = 0.02

	fish_filename = config_file.video_dir +\
		config_file.video_name + "_posture_fish" + str(count) + ".npz"

	# load the npz file of the given fish
	npz = np.load(fish_filename)

	outlines = npz["outline_points"] * cm_per_pixel

	outline_lengths = npz["outline_lengths"]
	#midline_lengths = npz["midline_lengths"]

	offsets = npz["offset"] * cm_per_pixel

	posture_frames = npz["frames"]

	return outlines, outline_lengths, offsets, posture_frames

def load_position_file(config_file, count):

	fish_filename = config_file.video_dir +\
		config_file.video_name + "_fish" + str(count) + ".npz"

	# load the npz file of the given fish
	npz = np.load(fish_filename)

	# extract X, Y position data
	X = npz["X#wcentroid"]
	Y = npz["Y#wcentroid"]
	frames_array = npz["frame"]

	return X, Y, frames_array

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

def collate(config_file, fill_gaps=False):

	start_frame = np.inf
	end_frame = 0

	for i in range(config_file.fish_count):
		filename = config_file.video_dir +\
			config_file.video_name  + "_fish" + str(i) + ".npz"

		npz = np.load(filename)

		frames = npz["frame"]

		curr_end_frame = frames[-1]
		curr_start_frame = frames[0]

		end_frame = end_frame if end_frame > curr_end_frame else curr_end_frame
		start_frame = start_frame if start_frame < curr_start_frame else curr_start_frame

	start_frame = int(start_frame)
	end_frame = int(end_frame)
	total_frames = int(end_frame-start_frame+1)

	Xall = np.zeros((config_file.fish_count, total_frames)) + np.inf
	Yall = np.zeros((config_file.fish_count, total_frames)) + np.inf
	missing_all = np.zeros((config_file.fish_count, total_frames), dtype=bool) + 1

	for count in range(config_file.fish_count):
		filename = config_file.video_dir +\
			config_file.video_name  + "_fish" + str(count) + ".npz"

		X, Y, frame_array = load_position_file(config_file, count)

		npz = np.load(filename)
		missing = npz["missing"]

		if fill_gaps:
			X = fill_in_gaps(X)
			Y = fill_in_gaps(Y)

		Xall[count, int(frame_array[0])-start_frame:int(frame_array[-1])+1] = X
		Yall[count, int(frame_array[0])-start_frame:int(frame_array[-1])+1] = Y
		missing_all[count, int(frame_array[0])-start_frame:int(frame_array[-1])+1] = missing

	if fill_gaps:
		reject_frames = []
	else:
		reject_frames = np.zeros(total_frames, dtype=bool)

		for i in range(total_frames):
			if sum(missing_all[:,i]) > 3:
				reject_frames[i] = True

	return Xall, Yall, reject_frames

if __name__=="__main__":
	filename = sys.argv[1]

	cfg = Config("config_files/"+filename+".csv")

	Xall, Yall, _ = collate(cfg, fill_gaps=False)

	# Xall_new = np.zeros((Xall.shape[0]-1, Xall.shape[1]))
	# Yall_new = np.zeros((Xall.shape[0]-1, Xall.shape[1]))

	Xall_new = Xall[1:,:]
	Yall_new = Yall[1:,:]

	np.savez("posAll_"+filename+".npz", Xall=Xall_new, Yall=Yall_new)
	#read_csv()
	exit()
	#write_csv()
	#make_histogram()
