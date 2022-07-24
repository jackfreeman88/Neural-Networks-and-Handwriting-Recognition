'''kmeans.py
Performs K-Means clustering
Jack Freeman
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans():
	def __init__(self, data=None):
		'''KMeans constructor'''

		# k: int. Number of clusters
		self.k = None
		# centroids: ndarray. shape=(k, self.num_features)
		#	k cluster centers
		self.centroids = None
		# data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
		#	Holds index of the assigned cluster of each data sample
		self.data_centroid_labels = None

		# inertia: float.
		#	Mean squared distance between each data sample and its assigned (nearest) centroid
		self.inertia = None

		# data: ndarray. shape=(num_samps, num_features)
		self.data = data
		# num_samps: int. Number of samples in the dataset
		self.num_samps = None
		# num_features: int. Number of features (variables) in the dataset
		self.num_features = None
		if data is not None:
			self.num_samps, self.num_features = data.shape

	def set_data(self, data):
		'''Replaces data instance variable with `data`.'''

		
		self.data = data
		self.num_samps = data.shape[0]
		self.num_features = data.shape[1]

	def get_data(self):
		'''Get a COPY of the data	'''
	
		return self.data.copy()

	def get_centroids(self):
		'''Get the K-means centroids '''
		
		return self.centroids

	def get_data_centroid_labels(self):
		'''Get the data-to-cluster assignments '''
	   
		return self.data_centroid_labels

	def dist_pt_to_pt(self, pt_1, pt_2):
		'''Compute the Euclidean distance between data samples `pt_1` and `pt_2` '''
		
		return np.sqrt(np.sum((pt_1-pt_2)**2))

	def dist_pt_to_centroids(self, pt, centroids):
		'''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
		self.centroids'''
		
		return np.sqrt(np.sum((pt-centroids)**2,1))

	def initialize(self, k):
		'''Initializes K-means by setting the initial centroids (means) to K unique randomly
		selected data samples'''
		
		
		self.k = k
		idx = np.random.choice(np.arange(self.num_samps),k,replace=False)
		self.centroids = self.data[np.ix_(idx)]
		
		return self.centroids


	def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
		'''Performs K-means clustering on the data'''
		
		self.initialize(k)
		iter = 0
		diff = np.ones((self.k,self.num_features))
		while (iter <= max_iter and diff.any() > tol):
			self.update_labels(self.centroids)
			self.centroids, diff = self.update_centroids(self.k,self.data_centroid_labels,self.centroids)
			iter += 1
		self.compute_inertia()
		
		if (verbose == True):
			print(f'Iterations:{iter}')
			print(f'Difference in Centroid Locations between Iterations: {diff}')
			print(f'Centroid Locations: {self.centroids}')
			print(f'Labels: {labels}')
			print(f'Inertia: {self.inertia}')
		
		return self.inertia, iter

	def cluster_batch(self, k=2, n_iter=1, verbose=False):
		'''Run K-means multiple times, each time with different initial conditions.
		Keeps track of K-means instance that generates lowest inertia. Sets the following instance
		variables based on the best K-mean run:
		- self.centroids
		- self.data_centroid_labels
		- self.inertia'''
		
		low_inert = 10000000
		tot_iter = 0
		for i in range(n_iter):
			inert, num_iter = self.cluster(k)
			if (inert < low_inert):
				low_inert = inert
				centroids = self.centroids
				labels = self.data_centroid_labels
			tot_iter += num_iter		
		self.centroids = centroids
		self.data_centroid_labels = labels
		self.inertia = low_inert
		
		return tot_iter/n_iter

	def update_labels(self, centroids):
		'''Assigns each data sample to the nearest centroid'''
		
		labels = np.zeros(self.num_samps, dtype = int)
		for i in range(self.num_samps):
			distances = self.dist_pt_to_centroids(self.data[i],centroids)
			labels[i] = np.where(distances == np.amin(distances))[0][0]
		self.data_centroid_labels = labels
		
		return self.data_centroid_labels

	def update_centroids(self, k, data_centroid_labels, prev_centroids):
		'''Computes each of the K centroids (means) based on the data assigned to each cluster'''
		
		new_centroids = np.zeros((k,self.num_features))
		for i in range(k):
			if (i not in self.data_centroid_labels):
				new_centroids[i] = self.centroids[i]
			else:
				new_centroids[i] = np.mean(self.data[self.data_centroid_labels == i],0)
		centroid_diff = new_centroids - prev_centroids
		 
		return new_centroids, centroid_diff

	def compute_inertia(self):
		'''Mean squared distance between every data sample and its assigned (nearest) centroid'''
		
		error = 0
		for i in range(self.k):
			error += np.sum((self.data[self.data_centroid_labels==i]-self.centroids[i])**2)
		self.inertia = error/self.num_samps
		
		return self.inertia

	def plot_clusters(self):
		'''Creates a scatter plot of the data color-coded by cluster assignment.'''
		
		bold_10 = cartocolors.qualitative.Bold_10.mpl_colors
		for i in range(self.k):
			plt.scatter(self.data[self.data_centroid_labels == i][:,0], self.data[self.data_centroid_labels == i][:,1], marker = '.', color = bold_10[i], label = f'cluster {i}')
		plt.scatter(self.centroids[:,0],self.centroids[:,1], marker='o',color='black', label = 'centroids')
		plt.show()


	def elbow_plot(self, max_k):
		'''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.'''
		inertia = []
		k = np.arange(max_k) + 1
		for i in range(max_k):
			inertia.append(self.cluster(k[i])[0])
		
		plt.plot(k,inertia)
		plt.xlabel('k clusters')
		plt.ylabel('Inertia')
		plt.xticks(k)

	def replace_color_with_centroid(self):
		'''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
		Used with image compression after K-means is run on the image vector.'''
		
		copy = self.get_data()
		for i in range(self.k):
			copy[self.data_centroid_labels==i] = self.centroids[i]
		self.set_data(copy)
