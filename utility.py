import numpy as np
import csv
import matplotlib.pyplot as plt

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

def onclick(event):
	print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		('double' if event.dblclick else 'single', event.button,
		event.x, event.y, event.xdata, event.ydata))

def make_histogram(switch_array_1, switch_array_2, switch_array, plotcolor="blue"):
	del_t = []
	t = [0]

	for i in range(1,len(switch_array)):
		t.append(switch_array[i].frame_num)
		del_t.append(switch_array[i].frame_num - switch_array[i-1].frame_num)

	t.sort()


	#plt.figure(1)
	fig, ax = plt.subplots()
	ax.bar(t[1:],del_t,width=100, color=plotcolor)
	plt.title("ID Switch locations in video")
	plt.xlabel("Frame stamp")
	plt.ylabel("Length of errorless tracking interval")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	#plt.figure(2)
	fig, ax = plt.subplots()
	plt.bar(range(1,len(switch_array)),del_t, color=plotcolor)
	plt.title("Time differences between two switches")
	plt.ylabel("Length of errorless tracking intervals")
	plt.xlabel("ID Switch number")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	del_t.sort()
	print("Differences:")
	print(del_t)
	print("Framestamps:")
	print(t)
	print("Switches: ", len(switch_array)-2)
	plt.show()

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

	min_frames = np.inf

	for i in range(config_file.fish_count):
		filename = config_file.video_dir +\
			config_file.video_name  + "_fish" + str(i) + ".npz"

		npz = np.load(filename)

		curr_frames = len(npz["X"])
		min_frames = min_frames if min_frames < curr_frames else curr_frames

	Xall = np.zeros((config_file.fish_count, min_frames))
	Yall = np.zeros((config_file.fish_count, min_frames))

	for count in range(config_file.fish_count):
		filename = config_file.video_dir +\
			config_file.video_name  + "_fish" + str(count) + ".npz"

		X, Y, _ = load_position_file(config_file, count)

		if fill_gaps:
			X = fill_in_gaps(X)
			Y = fill_in_gaps(Y)
		Xall[count,:] = X[0:min_frames]
		Yall[count,:] = Y[0:min_frames]

	return Xall, Yall

if __name__=="__main__":
	read_csv()
	#write_csv()
	#make_histogram()
