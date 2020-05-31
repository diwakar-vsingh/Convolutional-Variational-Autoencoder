import matplotlib.pyplot as plt
import PIL
import imageio
import glob


def display_image(image_dir, epoch_no):
	return PIL.Image.open((image_dir + '/image_at_epoch_{:04d}.png').format(epoch_no))


def generate_and_save_images(model, epoch, test_input, image_dir):
	predictions = model.decode(test_input)
	plt.figure(figsize=(6, 6))
	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i, :, :, 0], cmap='gray')
		plt.axis('off')
	
	# tight_layout minimizes the overlap between 2 sub-plots
	plt.savefig((image_dir + '/image_at_epoch_{:04d}.png').format(epoch))
	plt.show()


def generate_gif(image_dir, anim_file):
	with imageio.get_writer(anim_file, mode='I') as writer:
		filenames = glob.glob(image_dir + '/image*.png')
		filenames = sorted(filenames)
		last = -1
		for i, filename in enumerate(filenames):
			frame = 2 * (i ** 0.5)
			if round(frame) > round(last):
				last = frame
			else:
				continue
			image = imageio.imread(filename)
			writer.append_data(image)
		image = imageio.imread(filename)
		writer.append_data(image)
