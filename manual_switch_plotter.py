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
	
	del_t_chained = []

	t_1 = []
	del_t_1 = []
	del_t_1_indices = []

	t_2 = []
	del_t_2 = []
	del_t_2_indices = []

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
		
		if switch_array[i].type == -1:
			del_t_chained.append(diff)
		else:
			if del_t_chained[-1] < 0:
				del_t_chained.append(diff)
			else:
				del_t_chained[-1] += diff

	del_t_1 = np.array(del_t_1)
	del_t_2 = np.array(del_t_2)

	framerate = 25

	fig, axis = plt.subplots()
	axis.bar(t_1, del_t_1/framerate, width=25, color="red", label="Type 0 and 1 errors")
	axis.bar(t_2, del_t_2/framerate, width=25, color="blue", label="Type 2 errors")
	plt.title("Manual Tracking:\nID Switch locations in video "+filename)
	plt.xlabel("Frame stamp")
	plt.ylabel("Length of errorless tracking interval (in s)")
	plt.legend(loc="upper right")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	fig, axis = plt.subplots()
	axis.bar(del_t_1_indices, del_t_1, color="red", label="Type 0 and 1 errors")
	axis.bar(del_t_2_indices, del_t_2, color="blue", label="Type 2 errors")
	plt.title("Manual Tracking:\nTime differences between two switches in video "+filename)
	plt.ylabel("Length of errorless tracking intervals")
	plt.xlabel("ID Switch number")
	plt.legend(loc="upper right")
	cid = fig.canvas.mpl_connect('button_press_event', onclick)

	fig, axis = plt.subplots()
	del_t = np.array(del_t)
	del_t = del_t[del_t>(framerate*5)]
	axis.hist(del_t/framerate, bins=40)
	plt.title("Manual Tracking:\nTime interval frequencies in video "+filename)
	plt.xlabel("Length of tracking interval in seconds")
	plt.ylabel("Frequency")
	
	fig, axis = plt.subplots()
	del_t_chained = np.array(del_t_chained)
	del_t_chained = del_t_chained[del_t_chained>(framerate*5)]
	axis.hist(del_t_chained/framerate, bins=40)
	plt.title("Manual Tracking:\nTime interval frequencies in video "+filename+" POST CORRECTION")
	plt.xlabel("Length of tracking interval in seconds")
	plt.ylabel("Frequency")

def analyze(filename):

	switch_array = utils.read_csv("csv_files/"+filename+"/manual_switches_"+filename+".csv")

	make_histogram(switch_array, filename)
	plt.show()

if __name__ == "__main__":
	filename = sys.argv[1]
	analyze(filename)
