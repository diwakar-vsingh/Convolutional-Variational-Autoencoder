import tensorflow as tf


class CVAE(tf.keras.Model):
	def __init__(self, batch_size, latent_dim):
		super(CVAE, self).__init__()
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.inference_net = self.encoder()
		self.inference_net.summary()
		self.generative_net = self.decoder()
		self.generative_net.summary()
	
	def encoder(self):
		inputs = tf.keras.Input(shape=(28, 28, 1))
		x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(inputs)
		x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(2 * self.latent_dim)(x)
		model = tf.keras.Model(inputs, x)
		return model
	
	def decoder(self):
		inputs = tf.keras.Input(shape=self.latent_dim)
		x = tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu)(inputs)
		x = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
		x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same',
		                                    activation='relu')(x)
		x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same',
		                                    activation='relu')(x)
		x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same')(x)
		model = tf.keras.Model(inputs, x)
		return model
	
	def encode(self, x):
		"""
		Function to feed images into encoder and encode the latent space
		"""
		
		# Encoder output
		encoder_output = self.inference_net(x)
		
		# Latent variable distribution parameters
		z_mean = encoder_output[:, :self.latent_dim]
		z_log_var = encoder_output[:, self.latent_dim:]
		
		return z_mean, z_log_var
	
	@tf.function
	def sample(self):
		"""
		Reparameterization trick by sampling from an isotropic unit Gaussian.
		Arguments: z_mean, z_log_var:
		Returns: z [sampled latent vector]
		"""
		eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)
	
	def reparameterize(self, z_mean, z_log_var):
		"""
		VAE reparameterization: given a mean and logsigma, sample latent variables
		"""

		epsilon = tf.random.normal(shape=z_mean.shape)
		z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
		return z
	
	def decode(self, z, apply_sigmoid=False):
		"""
		Use the decoder to output the reconstructed image
		"""
		reconstruction = self.generative_net(z)
		if apply_sigmoid:
			probs = tf.sigmoid(reconstruction)
			return probs
		return reconstruction
