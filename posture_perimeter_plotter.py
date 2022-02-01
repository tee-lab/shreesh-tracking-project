import numpy as np
import matplotlib.pyplot as plt
import utility as utils
import sys

posture_frames = np.array(1)
outlines = np.array(1)

outline_lengths = np.array(1)

outline_iterator = 0

X = 1
Y = 1

posture_frames = []
frames_array = []
current_fish_count = -1

cm_per_pixel = 0.02

def get_perims(config_file, count):

	outlines, outline_lengths, _, _ = utils.load_posture_file(config_file, count)

	outline_iterator = 0
	perims = np.zeros(outline_lengths.shape)

	for i in range(len(outline_lengths)):

		outline_iterator_next = outline_iterator + int(outline_lengths[i])

		outline_slice = outlines[outline_iterator:outline_iterator_next, :]

		perims[i] = utils.get_perimeter(outline_slice)

		outline_iterator = outline_iterator_next

	return perims

def get_skips(config_file, count):

	X, Y, _ = utils.load_position_file(config_file, count)

	X = utils.fill_in_gaps(X)
	Y = utils.fill_in_gaps(Y)

	return utils.get_distance(X[1:], Y[1:], X[0:-1], Y[0:-1])

def plot_perims(cfg):

	all_perims = np.array([])

	for count in range(cfg.fish_count):
		all_perims = np.append(all_perims, get_perims(cfg, count))

	plt.figure(1)
	plt.hist(all_perims, bins=40)
	plt.title("Perimeter frequencies")
	plt.xlabel("Perimeter (cm)")
	plt.ylabel("Frequency")

def plot_skips(cfg):

	all_skips = np.array([])

	for count in range(cfg.fish_count):
		all_skips = np.append(all_skips, get_skips(cfg, count))

	plt.figure(2)
	plt.hist(all_skips, bins=40)
	plt.title("Skips")
	plt.xlabel("Skips (cm)")
	plt.ylabel("Frequency")

if __name__ == "__main__":

	filename = sys.argv[1]

	cfg = utils.Config("config_files/"+filename+".csv")

	plot_perims(cfg)
	plot_skips(cfg)
	plt.show()

	#perimeter: 0.75
	#skip: 0.55
