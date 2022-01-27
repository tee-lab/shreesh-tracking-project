import numpy as np

def main():
	a = np.mgrid[0:5:0.5,0:5:0.5]
	b = a.reshape((2,100))
	b = b.T
	#a0 = a[0,:,:]
	#a1 = a[1,:,:]
	#b = np.concat(a,axis=2)
	print(b)
	print(a.shape)
	print(a[0,:,:],a[1,:,:])

if __name__ == "__main__":
	main()
