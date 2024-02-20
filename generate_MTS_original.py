import os
import random
import csv
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def get_individual_objects_translated(img):
    """Returns individual objects in separated images"""
    # Bad sample flag
    bad_sample = False
    # Find contours
    image1 = img.copy()
    image2 = img.copy()
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # this is to deal with different opencv versions
    # If there is a single contour return with flag
    if len(cnts) == 1:
        bad_sample = True
        return image1, image2, bad_sample
    else:
        # Delete objects
        cv2.drawContours(image1, cnts, 0, color=[255,255,255], thickness=-1)
        cv2.drawContours(image2, cnts, 1, color=[255,255,255], thickness=-1)
        return image1, image2, bad_sample

def get_individual_objects_centered(img):
    """Returns individual objects centered in separated images"""
    # Bad sample flag
    bad_sample = False
    # Find contours
    image1 = img.copy()
    image2 = img.copy()
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # this is to deal with different opencv versions
    
    # If there is a single contour return with flag
    if len(cnts) == 1:
        bad_sample = True
        return image1, image2, bad_sample
    else:
        # Center object 1
        moments1 = cv2.moments(cnts[0])
        cx1 = int(moments1['m10']/moments1['m00'])
        cy1 = int(moments1['m01']/moments1['m00'])
        dx1 = 64 - cx1
        dy1 = 64 - cy1
        t_matrix1 = np.float32([[1, 0, dx1], [0, 1, dy1]])
        cv2.drawContours(image1, cnts, 1, color=[255,255,255], thickness=-1) # Delete second object
        t_image1 = cv2.warpAffine(image1, t_matrix1, (128, 128), borderValue=(255,255,255))

        # Center object 2
        moments2 = cv2.moments(cnts[1])
        cx2 = int(moments2['m10']/moments2['m00'])
        cy2 = int(moments2['m01']/moments2['m00'])
        dx2 = 64 - cx2
        dy2 = 64 - cy2
        t_matrix2 = np.float32([[1, 0, dx2], [0, 1, dy2]])
        cv2.drawContours(image2, cnts, 0, color=[255,255,255], thickness=-1) # Delete first object
        t_image2 = cv2.warpAffine(image2, t_matrix2, (128, 128), borderValue=(255,255,255))

        # plt.imshow(t_image1)
        # plt.show()
        # plt.imshow(t_image2)
        # plt.show()
        return t_image1, t_image2, bad_sample

def translate_objects_to_MTS_positions(img):
	"""
	Takes a 'different' sample and repositions objects into 3 cannonical positios.
	Pos 1 is around (64, 32), pos 2 is around (32, 96), and pos 3 is around (96, 96).
	This function makes two samples, the first with the match to the left, 
	and the second with the match to the right.
	"""
	# Bad sample flag
	bad_sample = False
	# Find contours
	image1 = img.copy()
	image2 = img.copy()
	image3 = img.copy()
	image4 = img.copy()
	image5 = img.copy()
	image6 = img.copy()
	gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1] # this is to deal with different opencv versions

	# Sort contours
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][1], reverse=False))

	# Set possible translations to sample from
	pos_translations = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

	# If there is a single contour return with flag
	if len(cnts) == 1:
		bad_sample = True
		return image1, image2, bad_sample
	else:
		# First image
		while True:
			# Set object 1 at top pos
			moments1 = cv2.moments(cnts[0])
			cx1 = int(moments1['m10']/moments1['m00'])
			cy1 = int(moments1['m01']/moments1['m00'])
			tx = random.sample(pos_translations, 1)[0]
			ty = random.sample(pos_translations, 1)[0]
			dx1 = 64 + tx - cx1
			dy1 = 32 + ty - cy1
			t_matrix1 = np.float32([[1, 0, dx1], [0, 1, dy1]])
			cv2.drawContours(image1, cnts, 1, color=[255,255,255], thickness=-1) # Delete second object
			# image1 = cv2.circle(image1, (cx1,cy1), radius=0, color=(255, 0, 0), thickness=-1)
			t_image1 = cv2.warpAffine(image1, t_matrix1, (128, 128), borderValue=(255,255,255))

			# Set object 2 at left pos
			moments2 = cv2.moments(cnts[1])
			cx2 = int(moments2['m10']/moments2['m00'])
			cy2 = int(moments2['m01']/moments2['m00'])
			tx = random.sample(pos_translations, 1)[0]
			ty = random.sample(pos_translations, 1)[0]
			dx2 = 32 + tx - cx2
			dy2 = 96 + ty - cy2
			t_matrix2 = np.float32([[1, 0, dx2], [0, 1, dy2]])
			cv2.drawContours(image2, cnts, 0, color=[255,255,255], thickness=-1) # Delete first object
			# image2 = cv2.circle(image2, (cx2,cy2), radius=0, color=(255, 0, 0), thickness=-1)
			t_image2 = cv2.warpAffine(image2, t_matrix2, (128, 128), borderValue=(255,255,255))
			
			# Set object 1 at right pos
			moments1 = cv2.moments(cnts[0])
			cx1 = int(moments1['m10']/moments1['m00'])
			cy1 = int(moments1['m01']/moments1['m00'])
			tx = random.sample(pos_translations, 1)[0]
			ty = random.sample(pos_translations, 1)[0]
			dx1 = 96 + tx - cx1
			dy1 = 96 + ty - cy1
			t_matrix3 = np.float32([[1, 0, dx1], [0, 1, dy1]])
			cv2.drawContours(image3, cnts, 1, color=[255,255,255], thickness=-1) # Delete second object
			# image3 = cv2.circle(image3, (cx1,cy1), radius=0, color=(255, 0, 0), thickness=-1)
			t_image3 = cv2.warpAffine(image3, t_matrix3, (128, 128), borderValue=(255,255,255))
			
			# Get mask for second image
			gray2 = cv2.cvtColor(t_image2, cv2.COLOR_BGR2GRAY)
			thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

			# Merge images
			result1 = cv2.bitwise_and(t_image1, t_image1, mask=thresh2)
			result2 = cv2.bitwise_and(t_image2, t_image2, mask=255-thresh2)
			result_a = cv2.add(result1, result2)
			
			# Get mask for third image
			gray3 = cv2.cvtColor(t_image3, cv2.COLOR_BGR2GRAY)
			thresh3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

			# Merge images
			result1 = cv2.bitwise_and(result_a, result_a, mask=thresh3)
			result2 = cv2.bitwise_and(t_image3, t_image3, mask=255-thresh3)
			result_b = cv2.add(result1, result2)

			# Stop the iteration if there are exactly 3 contours
			gray_f = cv2.cvtColor(result_b, cv2.COLOR_BGR2GRAY)
			thresh_f = cv2.threshold(gray_f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
			cnts_f = cv2.findContours(thresh_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts_f = cnts_f[0] if len(cnts_f) == 2 else cnts_f[1] # this is to deal with different opencv versions
			
			# get bounds and check if they're touching edge
			height, width = thresh_f.shape[:2]
			touching_edge = [] # boolean array, index matches the contours list
			for con in cnts_f:
				# get bounds
				x, y, w, h = cv2.boundingRect(con)
				# check if touching edge
				on_edge = False
				if x <= 0 or (x + w) >= (width - 1):
					on_edge = True
				if y <= 0 or (y + h) >= (height - 1):
					on_edge = True
				# add to list
				touching_edge.append(on_edge)
			if not any(touching_edge) and len(cnts_f) == 3:
				break
		
		# Second image
		while True:
			# Set object 1 at top pos
			moments1 = cv2.moments(cnts[0])
			cx1 = int(moments1['m10']/moments1['m00'])
			cy1 = int(moments1['m01']/moments1['m00'])
			tx = random.sample(pos_translations, 1)[0]
			ty = random.sample(pos_translations, 1)[0]
			dx1 = 64 + tx - cx1
			dy1 = 32 + ty - cy1
			t_matrix6 = np.float32([[1, 0, dx1], [0, 1, dy1]])
			cv2.drawContours(image6, cnts, 1, color=[255,255,255], thickness=-1) # Delete second object
			# image1 = cv2.circle(image1, (cx1,cy1), radius=0, color=(255, 0, 0), thickness=-1)
			t_image6 = cv2.warpAffine(image6, t_matrix6, (128, 128), borderValue=(255,255,255))

			# Set object 1 at left pos
			moments1 = cv2.moments(cnts[0])
			cx1 = int(moments1['m10']/moments1['m00'])
			cy1 = int(moments1['m01']/moments1['m00'])
			tx = random.sample(pos_translations, 1)[0]
			ty = random.sample(pos_translations, 1)[0]
			dx1 = 32 + tx - cx1
			dy1 = 96 + ty - cy1
			t_matrix4 = np.float32([[1, 0, dx1], [0, 1, dy1]])
			cv2.drawContours(image4, cnts, 1, color=[255,255,255], thickness=-1) # Delete second object
			# image4 = cv2.circle(image4, (cx1,cy1), radius=0, color=(255, 0, 0), thickness=-1)
			t_image4 = cv2.warpAffine(image4, t_matrix4, (128, 128), borderValue=(255,255,255))

			# Set object 2 at right pos
			moments2 = cv2.moments(cnts[1])
			cx2 = int(moments2['m10']/moments2['m00'])
			cy2 = int(moments2['m01']/moments2['m00'])
			tx = random.sample(pos_translations, 1)[0]
			ty = random.sample(pos_translations, 1)[0]
			dx2 = 96 + tx - cx2
			dy2 = 96 + ty - cy2
			t_matrix5 = np.float32([[1, 0, dx2], [0, 1, dy2]])
			cv2.drawContours(image5, cnts, 0, color=[255,255,255], thickness=-1) # Delete first object
			# image5 = cv2.circle(image5, (cx1,cy1), radius=0, color=(255, 0, 0), thickness=-1)
			t_image5 = cv2.warpAffine(image5, t_matrix5, (128, 128), borderValue=(255,255,255))

			# Get mask for fourth image
			gray4 = cv2.cvtColor(t_image4, cv2.COLOR_BGR2GRAY)
			thresh4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

			# Merge images
			result1 = cv2.bitwise_and(t_image6, t_image6, mask=thresh4)
			result2 = cv2.bitwise_and(t_image4, t_image4, mask=255-thresh4)
			result_c = cv2.add(result1, result2)

			# Get mask for fith image
			gray5 = cv2.cvtColor(t_image5, cv2.COLOR_BGR2GRAY)
			thresh5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

			# Merge images
			result1 = cv2.bitwise_and(result_c, result_c, mask=thresh5)
			result2 = cv2.bitwise_and(t_image5, t_image5, mask=255-thresh5)
			result_d = cv2.add(result1, result2)
			
			# Stop the iteration if there are exactly 3 contours and none is touching the the edge
			gray_f = cv2.cvtColor(result_d, cv2.COLOR_BGR2GRAY)
			thresh_f = cv2.threshold(gray_f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
			cnts_f = cv2.findContours(thresh_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts_f = cnts_f[0] if len(cnts_f) == 2 else cnts_f[1] # this is to deal with different opencv versions
			
			# get bounds and check if they're touching edge
			height, width = thresh_f.shape[:2]
			touching_edge = [] # boolean array, index matches the contours list
			for con in cnts_f:
				# get bounds
				x, y, w, h = cv2.boundingRect(con)
				# check if touching edge
				on_edge = False
				if x <= 0 or (x + w) >= (width - 1):
					on_edge = True
				if y <= 0 or (y + h) >= (height - 1):
					on_edge = True
				# add to list
				touching_edge.append(on_edge)
			if any(touching_edge) == 0 and len(cnts_f) == 3:
				break

	return result_b, result_d, bad_sample

# Method for creating directory if it doesn't exist yet
def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def get_dataset(batch_size, tfrecord_dir, is_training=True, process_img=True):
    # Load dataset.
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_dir)
    
    # Define example reading function.
    def read_tfrecord(serialized_example):
        # Create a dictionary describing the features.
        feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'coordinates': tf.io.FixedLenFeature([], tf.string),
            'relative_position': tf.io.FixedLenFeature([], tf.int64)}
        
        # Parse example.
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Get image.
        image = tf.io.decode_png(example['image_raw'], channels=3)

        # Ensure shape dimensions are constant.
        image = tf.reshape(image, [128, 128, 3])

        # Preprocess image.
        if process_img:
            image = tf.cast(image, tf.float64)
            image /= 255.0
            # Sample-wise center image.
            mean = tf.reduce_mean(image)
            image -= mean
            # Sample-wise std normalization.
            std = tf.math.reduce_std(image)
            image /= std
        
        # Cast label to int64
        label = example['label']
        label = tf.cast(label, tf.int64)

        # Get coordinates.
        b_coors = example['coordinates']
        coors = tf.io.parse_tensor(b_coors, out_type=tf.float64) # restore 2D array from byte string
        coors = tf.reshape(coors, [4])

        # Cast relative position to int64
        rel_pos = example['relative_position']
        rel_pos = tf.cast(rel_pos, tf.int64)

        return image, (label, coors, rel_pos)
        
    # Parse dataset.
    dataset = raw_image_dataset.map(tf.autograph.experimental.do_not_convert(read_tfrecord))
    
    if is_training:
        # I'm shuffling the datasets at creation time, so this is no necesarry for now.
        dataset = dataset.shuffle(11200)
        # Infinite dataset to avoid the potential last partial batch in each epoch.
        # dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    
    return dataset

def make_datasets(save_dir, split):
	# Get tfrecord file
	dataset_path = f'data/original_{split}.tfrecords'
	parsed_dataset = get_dataset(
		batch_size=1,
		tfrecord_dir=dataset_path,
		is_training=True,
		process_img=False
		)
	# Make directories
	root_dir = f'{save_dir}'
	check_path(root_dir)
	split_dir = f'{root_dir}/{split}'
	check_path(split_dir)
	
	# Initialize csv file for image data
	ds_file = f"{root_dir}/{split}_annotations.csv"
	ds_file_header = ['ID', 'lr_label']
	
	# Open the file in write mode
	with open(ds_file, 'w') as f:
		# Create the csv writer
		writer = csv.writer(f)
		# Write header to the csv file
		writer.writerow(ds_file_header)
		# Iterate over tfrecord file
		counter = 0
		for data in parsed_dataset:
			# Get data
			img = data[0][0]
			img = img.numpy()
			img = np.squeeze(img)
			label = data[1][0][0].numpy()
			label = 1 - label

			# img = Image.fromarray(img)
			x1, x2, flag1 = translate_objects_to_MTS_positions(img)

			img_a = Image.fromarray(x1).convert('RGB')
			img_b = Image.fromarray(x2).convert('RGB')
			if label == 1:
				img_a.save(f'{split_dir}/{counter}.png')
				lr_label = 1
				row = [counter, lr_label]
				writer.writerow(row)
				counter += 1

				img_b.save(f'{split_dir}/{counter}.png')
				lr_label = 0
				row = [counter, lr_label]
				writer.writerow(row)
				counter += 1

if __name__ == '__main__':
	# Root directory
	save_dir = './data/svrt1_mts'
	check_path(save_dir)
	# Splits
	splits = ['train', 'val', 'test']
	# Make datasets
	for split in splits:
		make_datasets(save_dir, split)

	print('All datasets created!')