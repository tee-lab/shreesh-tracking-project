import posture_perimeter_analysis as perim
import position_analysis as posit
import utility as utils
import matplotlib.pyplot as plt
import numpy as np
import sys

def onclick(event):
	print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		('double' if event.dblclick else 'single', event.button,
		event.x, event.y, event.xdata, event.ydata))

def make_histogram(switch_array, removed_switches, filename):

	del_t = []

	t_1 = []
	del_t_1 = []
	del_t_1_indices = []

	t_2 = []
	del_t_2 = []
	del_t_2_indices = []

	removed_indices = []

	for i in range(1,len(switch_array)):

		diff = switch_array[i].frame_num - switch_array[i-1].frame_num
		diff = -diff if switch_array[i].type == -1 else diff

		if switch_array[i].type == 2:
			t_2.append(switch_array[i].frame_num)
			del_t_2.append(diff)
			del_t_2_indices.append(i)
		else:
			t_1.append(switch_array[i].frame_num)
			del_t_1.append(diff)
			del_t_1_indices.append(i)

		del_t.append(diff)

	for i in range(len(removed_switches)):
		removed_indices.append(removed_switches[i].frame_num)

	del_t_1 = np.array(del_t_1)
	del_t_2 = np.array(del_t_2)

	framerate = 25

	fig, axis = plt.subplots()
	axis.bar(t_1, del_t_1/framerate, width=25, color="red", label="Type 0 and 1 errors")
	axis.bar(t_2, del_t_2/framerate, width=25, color="blue", label="Type 2 errors")
	axis.bar(removed_indices, -2, width=25, color="green", label="Removed errors")
	plt.title("ID Switch locations in video "+filename)
	plt.xlabel("Frame stamp")
	plt.ylabel("Length of errorless tracking interval (in s)")
	plt.legend(loc="upper right")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	fig, axis = plt.subplots()
	axis.bar(del_t_1_indices, del_t_1, color="red", label="Type 0 and 1 errors")
	axis.bar(del_t_2_indices, del_t_2, color="blue", label="Type 2 errors")
	plt.title("Time differences between two switches in video "+filename)
	plt.ylabel("Length of errorless tracking intervals")
	plt.xlabel("ID Switch number")
	plt.legend(loc="upper right")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	fig, axis = plt.subplots()
	del_t = np.array(del_t)
	del_t = del_t[del_t>(framerate*5)]
	axis.hist(del_t/framerate, bins=40)
	plt.title("Time interval frequencies in video "+filename)
	plt.xlabel("Length of tracking interval in seconds")
	plt.ylabel("Frequency")

def analyze(filename, to_collate_to_csv):

	if to_collate_to_csv:
		switch_array_1 = utils.read_csv("csv_files/"+filename+"/posit_switches_"+filename+".csv")
		switch_array_2 = utils.read_csv("csv_files/"+filename+"/perim_switches_"+filename+".csv")

		switch_array = []
		for switch in switch_array_1:
			switch_array.append(switch)

		for switch in switch_array_2:
			switch_array.append(switch)

		switch_array.sort(key = lambda x: x.frame_num)

		utils.write_csv(switch_array, "csv_files/"+filename+"/all_switches_"+filename+".csv")

	else:
		switch_array = utils.read_csv("csv_files/"+filename+"/all_switches_"+filename+".csv")

	removed_switches = utils.read_csv("csv_files/"+filename+"/removed_perim_switches_"+filename+".csv")

	make_histogram(switch_array, removed_switches, filename)
	plt.show()

if __name__ == "__main__":
	filename = sys.argv[1]
	to_collate_to_csv = sys.argv[2] == "yes" or sys.argv[2] == "y"
	analyze(filename, to_collate_to_csv)
