import time
import os
import IPython
import datetime

from IPython import display
from VAE import *
from input_data import *
from loss_fn import *
from utility import *

# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

if __name__ == "__main__":
	with strategy.scope():
		# Hyper-parameters
		learning_rate = 1e-3
		latent_dim = 100
		epochs = 2
		num_examples_to_generate = 16
	
		# keeping the random vector constant for generation (prediction) so
		# it will be easier to see the improvement.
		random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
	
		# Instantiate dataset
		BATCH_SIZE_PER_REPLICA = 100
		GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
		dataset = dataset_fn(strategy, GLOBAL_BATCH_SIZE)
		(train_dist_dataset, test_dist_dataset) = dataset.create_batches()
	
		# Instantiate VAE model
		model = CVAE(GLOBAL_BATCH_SIZE, latent_dim)
	
		# Define optimizer
		optimizer = tf.keras.optimizers.Adam(learning_rate)
	
		# Create a checkpoint directory to store the checkpoints.
		checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
		checkpoint_dir = './training_checkpoints'
		checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		
		# Create image directory
		image_dir = './images'
		try:
			os.mkdir(image_dir)
		except:
			pass
		generate_and_save_images(model, 0, random_vector_for_generation, image_dir)
		
		# Set up summary writers to write the summaries to disk in different logs directory
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
		test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		test_summary_writer = tf.summary.create_file_writer(test_log_dir)
		
		loss_fn = calculate_losses(model, strategy, optimizer, GLOBAL_BATCH_SIZE, kl_weight=0.001)
		for epoch in range(1, epochs + 1):
			# TRAIN LOOP
			total_loss = 0.0
			num_batches = 0
			start_time = time.time()
			for train_x in train_dist_dataset:
				total_loss += loss_fn.distributed_train_step(train_x)
				num_batches += 1
			train_loss = total_loss / num_batches
			end_time = time.time()
			
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', train_loss, step=epoch)
			
			# TEST LOOP
			for test_x in test_dist_dataset:
				loss_fn.distributed_test_step(test_x)
			elbo = -loss_fn.test_loss.result()
			with test_summary_writer.as_default():
				tf.summary.scalar('loss', loss_fn.test_loss.result(), step=epoch)
		
			display.clear_output(wait=False)
			template = 'Epoch {}, Loss: {}, Test Loss: {}, Time elapsed for current epoch: {}'
			print(template.format(epoch + 1, train_loss, elbo, end_time - start_time))
			generate_and_save_images(model, epoch, random_vector_for_generation, image_dir)
			
			# Reset metrics every epoch
			loss_fn.test_loss.reset_states()
		
			if epoch % 2 == 0:
				checkpoint.save(checkpoint_prefix)
		
	plt.imshow(display_image(image_dir, epochs))
	plt.axis('off')  # Display images

	# Generate a GIF of all the saved images.
	anim_file = 'cvae.gif'
	generate_gif(image_dir, anim_file)

	if IPython.version_info >= (6, 2, 0, ''):
		display.Image(filename=anim_file)
