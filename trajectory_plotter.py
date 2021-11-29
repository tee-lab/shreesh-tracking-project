import matplotlib.pyplot as plt
import numpy as np

def reject_frames():
	
	global Xall, Yall	
	
	for frame in range(Xall.shape[1]):
		inactive_fish_count = 0
		for fish in range(Xall.shape[0]):
			if Xall[fish, frame] == np.inf:
				print("found inactive")
				inactive_fish_count += 1
		
		if inactive_fish_count > 0:
			Xall[:, frame] = np.inf
			Yall[:, frame] = np.inf
			print("Inactivated all fish at frame: ", frame)

npz = np.load("posAll_15.npz")

Xall = npz["Xall"]
Yall = npz["Yall"]

reject_frames()

plt.figure(1)
for i in range(Xall.shape[0]):
	plt.plot(Xall[i,:])
plt.title("all X")
plt.figure(2)
for i in range(Yall.shape[0]):
	plt.plot(Yall[i,:])
plt.title("all Y")
plt.figure(3)
for i in range(Yall.shape[0]):
	plt.plot(Xall[i,:], Yall[i,:])
plt.title("X vs Y")
plt.show()
