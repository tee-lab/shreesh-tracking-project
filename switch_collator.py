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

def make_histogram(switch_array):

	t_1 = []
	del_t_1 = []
	del_t_1_indices = []

	t_2 = []
	del_t_2 = []
	del_t_2_indices = []

	for i in range(1,len(switch_array)):

		diff = switch_array[i].frame_num - switch_array[i-1].frame_num

		if switch_array[i].type == 2:
			t_2.append(switch_array[i].frame_num)
			del_t_2.append(diff)
			del_t_2_indices.append(i)
		else:
			t_1.append(switch_array[i].frame_num)
			del_t_1.append(-diff if switch_array[i].type == -1 else diff)
			del_t_1_indices.append(i)

	fig, axis = plt.subplots()
	axis.bar(t_1, del_t_1, width=25, color="red", label="Type 1 errors")
	axis.bar(t_2, del_t_2, width=25, color="blue", label="Type 2 errors")
	plt.title("ID Switch locations in video")
	plt.xlabel("Frame stamp")
	plt.ylabel("Length of errorless tracking interval")
	plt.legend(loc="upper right")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	fig, axis = plt.subplots()
	axis.bar(del_t_1_indices, del_t_1, color="red", label="Type 1 errors")
	axis.bar(del_t_2_indices, del_t_2, color="blue", label="Type 2 errors")
	plt.title("Time differences between two switches")
	plt.ylabel("Length of errorless tracking intervals")
	plt.xlabel("ID Switch number")
	plt.legend(loc="upper right")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

def analyze(filename, to_collate_to_csv):

	if to_collate_to_csv:
		switch_array_1 = utils.read_csv("csv_files/posit_switches_"+filename+".csv")
		switch_array_2 = utils.read_csv("csv_files/perim_switches_"+filename+".csv")

		switch_array = []
		for switch in switch_array_1:
			switch_array.append(switch)

		for switch in switch_array_2:
			switch_array.append(switch)

		switch_array.sort(key = lambda x: x.frame_num)

		utils.write_csv(switch_array, "csv_files/all_switches_"+filename+".csv")

	else:
		switch_array = utils.read_csv("csv_files/all_switches_"+filename+".csv")

	make_histogram(switch_array)
	plt.show()

if __name__ == "__main__":
	filename = sys.argv[1]
	to_collate_to_csv = sys.argv[2] == "yes"
	analyze(filename, to_collate_to_csv)
