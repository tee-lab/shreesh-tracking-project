#checker
import numpy as np
import utility as utils
import sys

filename = sys.argv[1]

cfg = utils.Config("config_files/"+filename+".csv")
cfg_old = utils.Config("config_files/"+filename+"_old.csv")

Xall, _, _, _ = utils.collate(cfg)
Xall_old, _, _, _ = utils.collate(cfg_old)

if Xall.shape != Xall_old.shape:
	print("Not same")
	exit()

count = 0
for fish in range(Xall.shape[0]):
	for frame in range(Xall.shape[1]):
		if Xall[fish, frame] != Xall_old[fish, frame]:
			#print("Not same")
			#print(Xall[:, frame])
			#print(Xall_old[:, frame])
			#print(fish, frame)
			 count += 1
			#exit()
print(count)
print("Same")