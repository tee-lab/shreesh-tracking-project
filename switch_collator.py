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

def get_switch_intervals(switch_array):
	del_t = []

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
	
	type1_data = [t_1, del_t_1, del_t_1_indices]
	type2_data = [t_2, del_t_2, del_t_2_indices]
	
	return del_t, type1_data, type2_data

def make_histogram(switch_array, filename):

	del_t, type1_data, type2_data = get_switch_intervals(switch_array)
	
	t_1 = type1_data[0]
	del_t_1 = type1_data[1]
	del_t_1_indices = type1_data[2]
	
	t_2 = type2_data[0]
	del_t_2 = type2_data[1]
	del_t_2_indices = type2_data[2]

	del_t_1 = np.array(del_t_1)
	del_t_2 = np.array(del_t_2)

	framerate = 25
	
	plt.rc('font', size=10) #controls default text size
	plt.rc('axes', titlesize=20) #fontsize of the title
	plt.rc('axes', labelsize=20) #fontsize of the x and y labels
	plt.rc('xtick', labelsize=13) #fontsize of the x tick labels
	plt.rc('ytick', labelsize=13) #fontsize of the y tick labels
	plt.rc('legend', fontsize=13) #fontsize of the legend

	fig, axis = plt.subplots()
	axis.bar(t_1, del_t_1/framerate, width=25, color="red", label="Type 0 and 1 errors")
	axis.bar(t_2, del_t_2/framerate, width=25, color="blue", label="Type 2 errors")
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

def analyze(filename, to_collate_to_csv, to_plot=True):

	if to_collate_to_csv:
		switch_array_posit = utils.read_csv("csv_files/"+filename+"/posit_switches_"+filename+".csv")
		switch_array_perim = utils.read_csv("csv_files/"+filename+"/perim_switches_"+filename+".csv")
		
		switch_array = utils.merge_switch_arrays(switch_array_posit, switch_array_perim)

		utils.write_csv(switch_array, "csv_files/"+filename+"/all_switches_"+filename+".csv")

	else:
		switch_array = utils.read_csv("csv_files/"+filename+"/all_switches_"+filename+".csv")

	removed_switches = utils.read_csv("csv_files/"+filename+"/removed_perim_switches_"+filename+".csv")
	
	literally_all_switches = utils.merge_switch_arrays(switch_array, removed_switches)
	
	if to_plot:
		make_histogram(literally_all_switches, filename)
		make_histogram(switch_array, filename)
		plt.show()
	else:
		return get_switch_intervals(literally_all_switches)[0], get_switch_intervals(switch_array)[0]

if __name__ == "__main__":
	filename = sys.argv[1]
	to_collate_to_csv = sys.argv[2] == "yes" or sys.argv[2] == "y"
	analyze(filename, to_collate_to_csv)
