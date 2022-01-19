import posture_perimeter_analysis as perim
import position_analysis as posit
import utility as utils
import matplotlib.pyplot as plt
import numpy as np

def analyze():
	if True:
		switch_array_1 = utils.read_csv("posit_switches_2.csv")
		switch_array_2 = utils.read_csv("perim_switches_2.csv")

		switch_array = []
		for switch in switch_array_1:
			switch_array.append(switch)

		for switch in switch_array_2:
			switch_array.append(switch)

		switch_array.sort(key = lambda x: x.frame_num)

	else:
		switch_array = utils.read_csv("all_switches.csv")

	utils.make_histogram([], [], switch_array)

if __name__ == "__main__":
	analyze()
