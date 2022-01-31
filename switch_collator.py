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

def make_histogram(switch_array, axis=None, type="by_frame"):
	del_t = []
	t = [0]

	for i in range(1,len(switch_array)):
		diff = switch_array[i].frame_num - switch_array[i-1].frame_num
		t.append(switch_array[i].frame_num)
		if switch_array[i].type == -1:
			del_t.append(-diff)
		else:
			del_t.append(diff)

	t.sort()

	if axis == None:
		if type=="by_frame":
			fig, axis = plt.subplots()
			axis.bar(t[1:],del_t,width=25, color="blue")
			plt.title("ID Switch locations in video")
			plt.xlabel("Frame stamp")
			plt.ylabel("Length of errorless tracking interval")
		else:
			fig, axis = plt.subplots()
			axis.bar(range(1,len(switch_array)),del_t, color="blue")
			plt.title("Time differences between two switches")
			plt.ylabel("Length of errorless tracking intervals")
			plt.xlabel("ID Switch number")

		cid = fig.canvas.mpl_connect('button_press_event', onclick)

	else:
		if type=="by_frame":
			axis.bar(t[1:],del_t,width=25, color="red")
		else:
			axis.bar(range(1,len(switch_array)),del_t, color="red")
	# #plt.figure(2)

	return axis

def analyze(filename):
	switch_array_1 = utils.read_csv("csv_files/posit_switches_"+filename+".csv")
	switch_array_2 = utils.read_csv("csv_files/perim_switches_"+filename+".csv")

	switch_array = []
	for switch in switch_array_1:
		switch_array.append(switch)

	for switch in switch_array_2:
		switch_array.append(switch)

	switch_array.sort(key = lambda x: x.frame_num)

	utils.write_csv(switch_array, "csv_files/all_switches_"+filename+".csv")
	#switch_array = utils.read_csv("csv_files/all_switches_"+filename+".csv")

	ax1 = make_histogram(switch_array_1, type="by_frame")
	make_histogram(switch_array_2, axis=ax1, type="by_frame")

	ax2 = make_histogram(switch_array_1, type="by_index")
	make_histogram(switch_array_2, axis=ax2, type="by_index")

	make_histogram(switch_array, type="by_frame")
	make_histogram(switch_array, type="by_index")
	plt.show()

if __name__ == "__main__":
	filename = sys.argv[1]
	analyze(filename)
