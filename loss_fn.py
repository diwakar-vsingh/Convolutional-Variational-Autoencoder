import tensorflow as tf


class calculate_losses:
	def __init__(self, model, strategy, optimizer, global_batch, kl_weight=0.001):
		self.model = model
		self.strategy = strategy
		self.optimizer = optimizer
		self.kl_weight = kl_weight
		self.batch_size = global_batch
		
		# Define metrics
		self.test_loss = tf.keras.metrics.Mean(name="test_loss")
	
	def compute_apply_gradients(self, x):
		with tf.GradientTape() as tape:
			loss = self.compute_loss(x)
	
		# Compute the gradients
		grad = tape.gradient(loss, self.model.trainable_variables)
	
		# Apply gradients to variables
		self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
		return loss

	def test_step(self, x):
		self.test_loss(self.compute_loss(x))
	
	# `run` replicates the provided computation and runs it
	# with the distributed input.
	@tf.function
	def distributed_train_step(self, dataset_inputs):
		per_replica_losses = self.strategy.run(self.compute_apply_gradients, args=(dataset_inputs,))
		return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

	@tf.function
	def distributed_test_step(self, dataset_inputs):
		return self.strategy.run(self.test_step, args=(dataset_inputs,))

	@tf.function
	def compute_loss(self, x):
		"""
		Function to calculate VAE loss given:
		"""
		# Compute z, z_mean and z_log_var
		z_mean, z_log_var = self.model.encode(x)
		z = self.model.reparameterize(z_mean, z_log_var)
		
		# Reshape x and x_logit
		x_shape = tf.shape(x)
		x = tf.reshape(x, [self.batch_size, x_shape[1] * x_shape[2] * x_shape[3]])
		
		x_logit = self.model.decode(z)
		x_logit_shape = tf.shape(x_logit)
		x_logit = tf.reshape(x_logit, [self.batch_size, x_logit_shape[1] * x_logit_shape[2] * x_logit_shape[3]])
		
		# KL divergence regularization loss.
		kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1.0 - z_log_var, axis=1)
		
		# Reconstruction loss:
		reconstruction_loss = tf.keras.losses.MSE(x, x_logit)
	
		# Total VAE loss
		vae_loss = self.kl_weight * kl_loss + reconstruction_loss
		return tf.nn.compute_average_loss(vae_loss, global_batch_size=self.batch_size)
