import numpy as np
import matplotlib.pyplot as plt
import utility as utils
import sys

def find_skips_radial(X, Y, threshold, switch_array, index, reject_frames):

	#this array holds index positions where ID skips have taken place
	skips = np.zeros(X.shape[0])

	skip_count = 0

	#scan X and Y vectors for skips

	for i in range(0, X.shape[0]-1):
		if reject_frames[i]:
			continue
		#first make sure finite values are being dealt with
		if X[i] != np.inf and X[i+1] != np.inf and Y[i] != np.inf and Y[i+1] != np.inf:
			#if motion in one frame is abnormally large, register as ID skip
			if np.sqrt((X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2) > threshold:
				switch = utils.IdSwitch()
				switch.set_type(1)
				switch.set_frame(i)
				switch.set_fish(index, -1)

				switch_array.append(switch)

	return switch_array

def find_inactivity_errors(switch_array, reject_frames):
	inside_a_gap = False

	for i in range(1,len(reject_frames)):
		if reject_frames[i]:
			if inside_a_gap:
				continue
			else:
				# add an error object that signifies the start of
				# inactivity interval
				switch = utils.IdSwitch()
				switch.set_type(0) #starts with 0
				switch.set_frame(i)
				switch.set_fish(-1, -1)

				switch_array.append(switch)

				inside_a_gap = True

		elif inside_a_gap:
			# add an error object that signifies the end of
			# inactivity interval
			switch = utils.IdSwitch()
			switch.set_type(-1) #ends with -1
			switch.set_frame(i-1)
			switch.set_fish(-1, -1)

			switch_array.append(switch)

			inside_a_gap = False

	return switch_array

def get_skip_count(skips, found_indices):
	count = 0
	for i in range(len(skips)):
		if i not in found_indices and skips[i] == 1:
			count += 1
			found_indices.append(i)

	return count, found_indices

def get_switches():

	filename = sys.argv[1]

	cfg = utils.Config("config_files/"+filename+".csv")

	max_fish_count = cfg.fish_count
	jump_threshold = cfg.jump_thresh

	Xall, _, reject_frames = utils.collate(cfg, fill_gaps=False)

	switch_array = []

	for count in range(0, max_fish_count):

		X, Y, frame_vec = utils.load_position_file(cfg, count)

		switch_array = find_skips_radial(X, Y, jump_threshold, switch_array, count, reject_frames)

		if False:
			#plot X position data
			plt.figure(1)
			plt.plot(frame_vec, X)
			#plt.plot(frame_vec[skips.astype(bool)], X[skips.astype(bool)])
			plt.plot(frame_vec, skips*20)
			plt.title("X position and skips vs Frames")
			plt.xlabel("Frame #")
			plt.ylabel("Xpos")

			#plot Y position data
			plt.figure(2)
			plt.plot(frame_vec, Y)
			plt.plot(frame_vec, skips*20)
			#plt.plot(frame_vec[skips.astype(bool)], Y[skips.astype(bool)])
			plt.title("Y position and skips vs Frames")
			plt.xlabel("Frame #")
			plt.ylabel("Ypos")

	switch_array = find_inactivity_errors(switch_array, reject_frames)

	switch_0 = utils.IdSwitch()
	switch_0.frame_num = 0
	switch_0.type = -2
	switch_last = utils.IdSwitch()
	switch_last.frame_num = Xall.shape[1]-1
	switch_last.type = -2
	switch_array.append(switch_0)
	switch_array.append(switch_last)

	print("Total skip count is: ", len(switch_array))
	print("Skips happened at frames:")
	switch_array.sort(key = lambda x: x.frame_num)
	for switch in switch_array:
		switch.display()

	utils.write_csv(switch_array, "csv_files/"+filename+"/posit_switches_"+filename+".csv")

if __name__ == "__main__":
	get_switches()
