import matplotlib.pyplot as plt
import numpy as np

npz = np.load("posAll.npz")

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
plt.show()