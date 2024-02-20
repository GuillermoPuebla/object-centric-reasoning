import os
import csv
import io
import math
import copy
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def check_path(path):
    """Method for creating directory if it doesn't exist yet."""
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
        dx1 = 32 - cx1
        dy1 = 64 - cy1
        t_matrix1 = np.float32([[1, 0, dx1], [0, 1, dy1]])
        cv2.drawContours(image1, cnts, 1, color=[255,255,255], thickness=-1) # Delete second object
        t_image1 = cv2.warpAffine(image1, t_matrix1, (128, 128), borderValue=(255,255,255))

        # Center object 2
        moments2 = cv2.moments(cnts[1])
        cx2 = int(moments2['m10']/moments2['m00'])
        cy2 = int(moments2['m01']/moments2['m00'])
        dx2 = 96 - cx2
        dy2 = 64 - cy2
        t_matrix2 = np.float32([[1, 0, dx2], [0, 1, dy2]])
        cv2.drawContours(image2, cnts, 0, color=[255,255,255], thickness=-1) # Delete first object
        t_image2 = cv2.warpAffine(image2, t_matrix2, (128, 128), borderValue=(255,255,255))

        # Put objects together
        bitwiseXor = cv2.bitwise_xor(t_image1, t_image2)
        bitwiseNot = cv2.bitwise_not(bitwiseXor)

        return bitwiseNot, bad_sample

def make_dataset(save_dir, split):
    """Builds single image version of svrt problem 1.
    Args:
        save_dir: directory to save generated images.
        split: 'test', 'train', or 'val'.
    Returns: nothing.
    """
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
    ds_file_header = ['ID', 'sd_label']
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
            row = [f'{counter}.png', label]
            # Save data
            img = Image.fromarray(img)
            img.save(f'{split_dir}/{counter}.png')
            writer.writerow(row)
            counter += 1


if __name__ == '__main__':
    # Root directory
    save_dir = './data/svrt1_sd'
    # Splits
    splits = ['train', 'val', 'test']
    # Make datasets
    for split in splits:
        make_dataset(save_dir, split)

    print('All all datasets created!')