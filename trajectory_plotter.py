import matplotlib.pyplot as plt
import numpy as np
import sys

npz = np.load("posAll_"+sys.argv[1]+".npz")

Xall = npz["Xall"]
Yall = npz["Yall"]

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

total_fish = np.sum(np.isfinite(Xall), axis=0)

plt.figure(4)
plt.plot(total_fish)
plt.title("Number of active fish per frame")

plt.show()
