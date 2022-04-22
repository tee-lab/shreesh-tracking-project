import utility as utils
import matplotlib.pyplot as plt
import numpy as np
import sys

def onclick(event):
	print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		('double' if event.dblclick else 'single', event.button,
		event.x, event.y, event.xdata, event.ydata))

def get_switch_intervals(switch_array):
	
	del_t = []
	
	del_t_chained = [0]

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
	
	type1_data = [t_1, del_t_1, del_t_1_indices]
	type2_data = [t_2, del_t_2, del_t_2_indices]
	
	overall_data = [del_t, del_t_chained]
	
	return overall_data, type1_data, type2_data

def make_histogram(switch_array, filename):

	overall_data, type1_data, type2_data = get_switch_intervals(switch_array)
	
	del_t = overall_data[0]
	del_t_chained = overall_data[1]
	
	t_1 = type1_data[0]
	del_t_1 = type1_data[1]
	del_t_1_indices = type1_data[2]
	
	t_2 = type2_data[0]
	del_t_2 = type2_data[1]
	del_t_2_indices = type2_data[2]
	
	framerate = 25
	
	del_t_1 = np.array(del_t_1)
	del_t_2 = np.array(del_t_2)
	
	plt.rc('font', size=10) #controls default text size
	plt.rc('axes', titlesize=20) #fontsize of the title
	plt.rc('axes', labelsize=20) #fontsize of the x and y labels
	plt.rc('xtick', labelsize=13) #fontsize of the x tick labels
	plt.rc('ytick', labelsize=13) #fontsize of the y tick labels
	plt.rc('legend', fontsize=13) #fontsize of the legend

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

def analyze(filename, to_plot=True):
	
	try:
		switch_array = utils.read_csv("csv_files/"+filename+"/manual_switches_"+filename+".csv")
	except:
		print("")
	else:
		if to_plot:
			make_histogram(switch_array, filename, False)
			plt.show()
		else:
			return get_switch_intervals(switch_array)

if __name__ == "__main__":
	filename = sys.argv[1]
	analyze(filename)
