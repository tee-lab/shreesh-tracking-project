# shreesh-tracking-project

All the code and files used for Shreesh's fish tracking project.

Using these scripts, we are trying to reduce the number of ID flips that are caused
by lapses in the tracking done by Trex. Two types of ID flips are observed in the data:

Type 1 skip:
	When a fish has been inactive for a long time, and its ID gets reassigned to
	another fish in the span of one frame, resulting in an abrupt, easy to identify
	jump in the position data.
Type 2 skip:
	When two (or more) fish come very close to each other and merge into one body.
	After separating, the two fish have their IDs flipped or shuffled around. Trex
	tries to account for such flips by itself, but predictably fails frequently.

By default, the scripts load the example data provided in the folder "data/".
To change this, download the rest of the position and posture data files from:
https://drive.google.com/drive/folders/1CP54dn0sA81RMlaLEoAEMmQaSeku9YXN?usp=sharing
and change the "max\_fish\_count" parameter to 30 in every script.

This is what each script accomplishes (as of the time of the commit):
	
	position\_analysis.py:
		Loads position (x, y) data of each fish and finds skips of type 1.
	
	posture\_area\_analysis.py:
		Loads the posture data of each fish and uses the posture area data to figure out when
		two or more fish merge into one body (ie trying to find ID flips of type 2).
		Performs minimal processing to figure out who merged with whom (this processing
		does not always give the right answer).
	
	posture\_perimeter\_analysis.py
		Loads the posture data of each fish and finds out the perimeter of each posture outline
		to figure out when two or more fish merge into one body (ie trying to find ID flips of
		type 2).
		Performs processing on the selected outlines to segment the merged body into two or
		more separate bodies. Two approaches are used:
			1) Use k-means clustering to find two clusters in the outline, and hence find the
			centroid of the resulting two clusters. These centroids will replace the old centroids
			in the position data found by Trex. Due to two different approaches used by my scripts
			and Trex (where I find cluster centroids obtained by the raw outline data, whereas
			Trex finds centroids using weights from the original segmented grayscale images of
			the fish in the video), this approach needs a bit more refining to work correctly.
			2) Find the curvature of the outline at each point to figure out where the two
			heads of the fish are (not very reliable).
