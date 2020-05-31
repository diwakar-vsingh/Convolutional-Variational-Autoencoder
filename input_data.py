import numpy as np
import tensorflow as tf


class dataset_fn:
	def __init__(self, strategy, BATCH_SIZE=100):
		
		self.strategy = strategy
		# Load the MNIST dataset
		print("Downloading MNIST dataset")
		(self.train_images, _), (self.test_images, _) = tf.keras.datasets.mnist.load_data()
		
		# Dataset parameters
		self.TRAIN_BUF = len(self.train_images)
		self.TEST_BUF = len(self.test_images)
		self.BATCH_SIZE = BATCH_SIZE
		
		# Pre-process the dataset
		print("Pre-Processing on the dataset")
		self.pre_processing()

	def pre_processing(self):
		# Add channel dimension and cast data to float32
		self.train_images = np.expand_dims(self.train_images, axis=-1).astype('float32')
		self.test_images = np.expand_dims(self.test_images, axis=-1).astype('float32')

		# Normalize the images to the range [0., 1.]
		self.train_images /= 255
		self.test_images /= 255

		# Binarization
		self.train_images[self.train_images >= 0.5] = 1
		self.train_images[self.train_images < 0.5] = 0
		self.test_images[self.test_images >= 0.5] = 1
		self.test_images[self.test_images < 0.5] = 0

	def create_batches(self):
		# Convert the dataset into tf.data.Dataset format
		self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images)
		self.test_dataset = tf.data.Dataset.from_tensor_slices(self.test_images)
		
		# Shuffle and batch the dataset
		self.train_dataset = self.train_dataset.shuffle(self.TRAIN_BUF).batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
		self.test_dataset = self.test_dataset.shuffle(self.TEST_BUF).batch(self.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
		
		# Distribute dataset
		self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
		self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)
		return self.train_dist_dataset, self.test_dist_dataset

