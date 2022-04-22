import numpy as np
import matplotlib.pyplot as plt
import utility as utils
import sys

def get_perims(config_file, count):

	outlines, outline_lengths, _, posture_frames = utils.load_posture_file(config_file, count)

	_, _, reject_frames, _ = utils.collate(config_file)

	outline_iterator = 0
	perims = np.zeros(outline_lengths.shape)

	for i in range(len(outline_lengths)):

		outline_iterator_next = outline_iterator + int(outline_lengths[i])

		if not (posture_frames[i] >= len(reject_frames) or reject_frames[posture_frames[i]]):
			outline_slice = outlines[outline_iterator:outline_iterator_next, :]
			perims[i] = utils.get_perimeter(outline_slice)
		else:
			perims[i] = 0

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

	all_perims = all_perims[all_perims>0]
	plt.figure(1)
	plt.hist(all_perims, bins=40)
	plt.title("Posture Perimeter frequencies")
	plt.xlabel("Perimeter (cm)")
	plt.ylabel("Frequency")

def plot_skips(cfg):

	all_skips = np.array([])

	for count in range(cfg.fish_count):
		all_skips = np.append(all_skips, get_skips(cfg, count))

	plt.figure(2)
	plt.hist(all_skips, bins=40)
	plt.title("Distance traveled per frame")
	plt.xlabel("Per-frame distance (cm)")
	plt.ylabel("Frequency")

if __name__ == "__main__":

	filename = sys.argv[1]

	cfg = utils.Config("config_files/"+filename+".csv")
	
	plt.rc('font', size=20) #controls default text size
	plt.rc('axes', titlesize=40) #fontsize of the title
	plt.rc('axes', labelsize=40) #fontsize of the x and y labels
	plt.rc('xtick', labelsize=26) #fontsize of the x tick labels
	plt.rc('ytick', labelsize=26) #fontsize of the y tick labels
	plt.rc('legend', fontsize=26) #fontsize of the legend

	plot_perims(cfg)
	plot_skips(cfg)
	plt.show()
