import utility as utils
import matplotlib.pyplot as plt
import numpy as np
import sys

def onclick(event):
	print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		('double' if event.dblclick else 'single', event.button,
		event.x, event.y, event.xdata, event.ydata))

def make_histogram(switch_array, filename):

	del_t = []

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

		del_t.append(diff)

	fig, axis = plt.subplots()
	axis.bar(t_1, del_t_1, width=25, color="red", label="Type 0 and 1 errors")
	axis.bar(t_2, del_t_2, width=25, color="blue", label="Type 2 errors")
	plt.title("ID Switch locations in video "+filename)
	plt.xlabel("Frame stamp")
	plt.ylabel("Length of errorless tracking interval")
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
	framerate = 25
	del_t = np.array(del_t)
	del_t = del_t[del_t>(framerate*5)]
	axis.hist(del_t/framerate, bins=40)
	plt.title("Time interval frequencies in video "+filename)
	plt.xlabel("Length of tracking interval in seconds")
	plt.ylabel("Frequency")

def analyze(filename):

	switch_array = utils.read_csv("csv_files/"+filename+"/manual_switches_"+filename+".csv")

	make_histogram(switch_array, filename)
	plt.show()

if __name__ == "__main__":
	filename = sys.argv[1]
	analyze(filename)
