import utility as utils
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import manual_switch_plotter as manual
import switch_collator

#os.chdir("csv_files")

filenames = os.listdir("./csv_files")

all_folders = []
for filename in filenames:
	if os.path.isdir(os.path.join(os.path.abspath("./csv_files"), filename)): # check whether the current object is a folder or not
		if len(filename) == 8:
			all_folders.append(filename)
			#print(filename)

pre_algo_del_t = []
post_algo_del_t = []
manual_del_t = []
manual_del_t_chained = []

for foldername in all_folders:
	filenames = os.listdir("csv_files/"+foldername+"/")
	
	if "manual_switches_"+foldername+".csv" in filenames:
		overall_data, _, _ = manual.analyze(foldername, to_plot=False)
		del_t = overall_data[0]
		del_t_chained = overall_data[1]
		
		for element in del_t:
			manual_del_t.append(element)
		for element in del_t_chained:
			manual_del_t_chained.append(element)
	
		del_t_pre_algo, del_t_post_algo = switch_collator.analyze(foldername, False, to_plot=False)
		
		for element in del_t_pre_algo:
			pre_algo_del_t.append(element)
		for element in del_t_post_algo:
			post_algo_del_t.append(element)

pre_algo_del_t = np.array(pre_algo_del_t)
post_algo_del_t = np.array(post_algo_del_t)
manual_del_t = np.array(manual_del_t)
manual_del_t_chained = np.array(manual_del_t_chained)

framerate = 25

#pre_algo_del_t = pre_algo_del_t[pre_algo_del_t>(framerate*5)]
#post_algo_del_t = post_algo_del_t[post_algo_del_t>(framerate*5)]
#manual_del_t = manual_del_t[manual_del_t>(framerate*5)]
#manual_del_t_chained = manual_del_t_chained[manual_del_t_chained>(framerate*5)]

pre_algo_del_t = pre_algo_del_t[pre_algo_del_t>=0]
post_algo_del_t = post_algo_del_t[post_algo_del_t>=0]
manual_del_t = manual_del_t[manual_del_t>=0]
manual_del_t_chained = manual_del_t_chained[manual_del_t_chained>=0]

#plt.rc('font', size=10) #controls default text size
#plt.rc('axes', titlesize=20) #fontsize of the title
#plt.rc('axes', labelsize=20) #fontsize of the x and y labels
#plt.rc('xtick', labelsize=13) #fontsize of the x tick labels
#plt.rc('ytick', labelsize=13) #fontsize of the y tick labels
#plt.rc('legend', fontsize=13) #fontsize of the legend

fig, axis = plt.subplots()

axis.hist(pre_algo_del_t/framerate, bins=40)
plt.title("Time interval frequencies from all videos \n PRE ALGORITHM")
plt.xlabel("Length of tracking interval in seconds")
plt.ylabel("Frequency")

fig, axis = plt.subplots()
axis.hist(post_algo_del_t/framerate, bins=40)
plt.title("Time interval frequencies from all videos \n POST ALGORITHM")
plt.xlabel("Length of tracking interval in seconds")
plt.ylabel("Frequency")

fig, axis = plt.subplots()
axis.hist(manual_del_t/framerate, bins=40)
plt.title("Time interval frequencies from all videos \n MANUALLY TRACKED")
plt.xlabel("Length of tracking interval in seconds")
plt.ylabel("Frequency")

fig, axis = plt.subplots()
axis.hist(manual_del_t_chained/framerate, bins=40)
plt.title("Time interval frequencies from all videos \n MANUALLY TRACKED AND CORRECTED")
plt.xlabel("Length of tracking interval in seconds")
plt.ylabel("Frequency")

fig, axis = plt.subplots(2,2, sharex=True,sharey=True)
axis[0,0].hist(pre_algo_del_t/framerate, bins=40)
axis[0,0].set_title("Pre Algorithm Frequencies")
axis[0,1].hist(post_algo_del_t/framerate, bins=40)
axis[0,1].set_title("Post Algorithm Frequencies")
axis[1,0].hist(manual_del_t/framerate, bins=40)
axis[1,0].set_title("Manual Tracking Frequencies") 
axis[1,1].hist(manual_del_t_chained/framerate, bins=40)
axis[1,1].set_title("Manual Tracking with Correction Frequencies")

for ax in axis.flat:
    ax.set(xlabel='Frequency', ylabel='Length of tracking interval in seconds')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axis.flat:
    ax.label_outer()

plt.show()