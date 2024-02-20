import os
import csv
import math
import copy
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Core graphical functions
def regular_polygon(sides, radius=10, rotation=0, translation=None):
	"""Calculates the vertices of a regular polygon by sweeping out a circle, and puting n equally spaced points on it."""
	# The first thing to do is work out the angle (in radians) of each wedge from the center outwards.
	# The total number of radians in a circle is 2 pi, so our value is 2 pi / n per segment.
	one_segment = math.pi * 2 / sides
	# After that a bit of basic trig gives us our points. At this point we scale by our desired radius,
	# and have the opportunity to offset the rotation by a fixed amount too.
	points = [
		(int(math.sin(one_segment * i + rotation) * radius),
		int(math.cos(one_segment * i + rotation) * radius))
		for i in range(sides)]

	original_points = copy.copy(points)
	# After that we translate the values by a certain amount, because you probably want your polygon
	# in the center of the screen, not in the corner.
	if translation:
		points = [[sum(pair) for pair in zip(point, translation)]
				for point in points]
	return points, original_points

def ccw_sort(polygon_points):
	"""Sort the points counter clockwise around the mean of all points. The sorting can be imagined like a
	radar scanner, points are sorted by their angle to the x axis."""
	polygon_points = np.array(polygon_points)
	mean = np.mean(polygon_points,axis=0)
	d = polygon_points-mean
	s = np.arctan2(d[:,0], d[:,1])
	return polygon_points[np.argsort(s),:]

def iregular_polygon_from_regular(sides, radius=1, rotation=0, translation=None, max_dev=0):
	# Get regular polygon.
	points, original_points = regular_polygon(sides=sides, radius=radius, rotation=rotation, translation=translation)

	# Add noise.
	noise = [[np.random.randint(-max_dev, max_dev+1), np.random.randint(-max_dev, max_dev+1)] for x in points]
	points = [[x[0] + y[0], x[1] + y[0]] for x, y in zip(points, noise)]
	original_points = [[x[0] + y[0], x[1] + y[0]] for x, y in zip(original_points, noise)]
	
	# Return points and cero-centerd points.
	return ccw_sort(points), ccw_sort(original_points)

def divide_polygon(points):
	"""Divides polygon at the midsection of every side.
	Args:
		points: list of points.
	Returns:
		List of lits of points."""
	mid_points = []
	for i in range(len(points)):
		if i == len(points) - 1:
			midpoint = [(points[i][0] + points[0][0]) / 2, (points[i][1] + points[0][1]) / 2]
		else:
			midpoint = [(points[i][0] + points[i+1][0]) / 2, (points[i][1] + points[i+1][1]) / 2]
		mid_points.append(midpoint)

	new_points = []
	for i in range(len(mid_points)):
		if i == len(mid_points) - 1:
			new_points.append([mid_points[i], points[i], points[0]])
		else:
			new_points.append([mid_points[i], points[i], points[i+1]])

	return new_points

def displace_line_around_origin(point_list, d):
	"""Displace a line (list of points) away from the center (0, 0) d units."""
	point = point_list[1]
	x, y = point
	d_x = d if x >= 0 else -d
	d_y = d if y >= 0 else -d
	displacement = [d_x, d_y]

	displaced_point_list = [[sum(pair) for pair in zip(point, displacement)] for point in point_list]
	
	return displaced_point_list

def displace_polygon_vertices(list_of_points, radius):
	"""Displace polygon subseccions randomly around the center.
	The displacement keeps the angles of the original polygon.
	This function assumes that points are the original polygon
	points around the coordinate (0,0).
	
	Args:
		points: list of points.
	Returns:
		List of lits of points."""
	
	mid_points = []
	for i in range(len(list_of_points)):
		if i == len(list_of_points) - 1:
			midpoint = [(list_of_points[i][0] + list_of_points[0][0]) / 2, (list_of_points[i][1] + list_of_points[0][1]) / 2]
		else:
			midpoint = [(list_of_points[i][0] + list_of_points[i+1][0]) / 2, (list_of_points[i][1] + list_of_points[i+1][1]) / 2]
		mid_points.append(midpoint)

	new_points = []
	for i in range(len(mid_points)):
		if i == len(mid_points) - 1:
			new_points.append([mid_points[i], list_of_points[0], mid_points[0]])
		else:
			new_points.append([mid_points[i], list_of_points[i+1], mid_points[i+1]])

	# All posible displacements to sample from.
	all_d = list(range(0, radius))
	random.shuffle(all_d)

	# Displace the points from the distance a randomly chosen amount.
	displaced_points = []
	counter = 0
	for point_list in new_points:
		d = all_d[counter]  # random.sample(all_d, 1)[0]
		new_point_list = displace_line_around_origin(point_list, d)
		displaced_points.append(new_point_list)
		counter += 1
		# Reset the counter if reach the end of all displacements.
		if counter >= len(all_d) - 1:
			counter = 0
	
	return displaced_points

def scramble_poligon(img, midpoint, radius):
	# Augment the radius to cover all pixels in teh target patch. 
	radius += 1
	# Get start points and end points fo the 4 quadrants.
	sp_1 = (midpoint[0]-radius, midpoint[1]-radius)
	ep_1 = midpoint

	sp_2 = (midpoint[0], midpoint[1]-radius)
	ep_2 = (midpoint[0]+radius, midpoint[1])

	sp_3 = (midpoint[0]-radius, midpoint[1])
	ep_3 = (midpoint[0], midpoint[1]+radius)

	sp_4 = midpoint
	ep_4 = (midpoint[0]+radius, midpoint[1]+radius)

	# Sample offsets.
	off_x = random.sample(list(range(0, int(radius/2))), 4)
	off_y = random.sample(list(range(0, int(radius/2))), 4)
	
	# Add offsets.
	new_sp_1 = (sp_1[0]-off_x[0], sp_1[1]-off_y[0])
	new_ep_1 = (ep_1[0]-off_x[0], ep_1[1]-off_y[0])

	new_sp_2 = (sp_2[0]+off_x[1], sp_2[1]-off_y[1])
	new_ep_2 = (ep_2[0]+off_x[1], ep_2[1]-off_y[1])

	new_sp_3 = (sp_3[0]-off_x[2], sp_3[1]+off_y[2])
	new_ep_3 = (ep_3[0]-off_x[2], ep_3[1]+off_y[2])

	new_sp_4 = (sp_4[0]+off_x[3], sp_4[1]+off_y[3])
	new_ep_4 = (ep_4[0]+off_x[3], ep_4[1]+off_y[3])
	
	# Copy patches.
	patch_1 = np.copy(img[sp_1[1]:ep_1[1], sp_1[0]:ep_1[0]])
	patch_2 = np.copy(img[sp_2[1]:ep_2[1], sp_2[0]:ep_2[0]])
	patch_3 = np.copy(img[sp_3[1]:ep_3[1], sp_3[0]:ep_3[0]])
	patch_4 = np.copy(img[sp_4[1]:ep_4[1], sp_4[0]:ep_4[0]])

	# Wipe out patches in img.
	img[sp_1[1]:ep_1[1], sp_1[0]:ep_1[0]] = (255, 255, 255)
	img[sp_2[1]:ep_2[1], sp_2[0]:ep_2[0]] = (255, 255, 255)
	img[sp_3[1]:ep_3[1], sp_3[0]:ep_3[0]] = (255, 255, 255)
	img[sp_4[1]:ep_4[1], sp_4[0]:ep_4[0]] = (255, 255, 255)

	# Paste patches in new locations.
	img[new_sp_1[1]:new_ep_1[1], new_sp_1[0]:new_ep_1[0]] = patch_1
	img[new_sp_2[1]:new_ep_2[1], new_sp_2[0]:new_ep_2[0]] = patch_2
	img[new_sp_3[1]:new_ep_3[1], new_sp_3[0]:new_ep_3[0]] = patch_3
	img[new_sp_4[1]:new_ep_4[1], new_sp_4[0]:new_ep_4[0]] = patch_4

	return img

def svrt_1_points(
	category=1, 
	radii=None, 
	sides=None, 
	rotations=None, 
	regular=False,
	irregularity=0.25, 
	displace_vertices=False
	):
	"""Returns polygon points for a single instance of a MTS problem.
	Args:
		category: 0 (left) or 1 (right).
		radii: radii of the base regular polygon. 2-tuple 8 to 14.
		sides: number of sides of the base regular polygon. 2-tuple 4 to 8.
		rotations: rotations of the polygons. 2-tuple 4 to 8.
		regular: whether to build regular or irregular polygons in radiants. 2-tuple form 0 to pi.
		irregularity: maximum level of random point translation for irregular polygons.
		displace_vertices: if True displaces second polygon subseccions randomly around its center in the positive cases.
	Returns:
		Two lists of polygon points."""

	# Polygon parameters.
	if radii is None:
		if displace_vertices:
			radius_1 = np.random.randint(10, 18)
			radius_2 = radius_1 #if category==1 else np.random.randint(10, 14)
		else:
			radius_1 = np.random.randint(10, 40) # np.random.randint(10, 14)
			radius_2 = radius_1 #if category==1 else np.random.randint(10, 40)
	else:
		radius_1, radius_2 = radii

	if sides is None:
		if displace_vertices:
			possible_sides = random.sample(list(range(3, 8)), 2)
		else:
			possible_sides = random.sample(list(range(3, 8)), 2)
		sides_1 = possible_sides[0]
		sides_2 = possible_sides[1]

	if rotations is None:
		rotation_1 = math.radians(random.randint(0, 360))
		rotation_2 = math.radians(random.randint(0, 360))

	# I need to calculate min_dev_1 based on the actual points not based on the maximum posible enclosing circle...
	
	if not regular and irregularity is None:
		max_dev_factor = np.random.choice([0.3, 0.4, 0.5, 0.6])
	else:
		max_dev_factor = irregularity
	max_dev_1 = int(radius_1 * max_dev_factor)
	min_dev_1 = radius_1 + max_dev_1
	max_dev_2 = int(radius_2 * max_dev_factor)
	min_dev_2 = radius_2 + max_dev_2

	# Positions.
	translation_1 = [np.random.randint(58, 70), np.random.randint(26, 38)]
	translation_2 = [np.random.randint(26, 38), np.random.randint(90, 102)]
	translation_3 = [np.random.randint(90, 102), np.random.randint(90, 102)]

	# Generate points.
	if category == 0 and regular and not displace_vertices:
		# A math.pi/4 (45 degrees) rotation gives the most stable polygons in the "1" category.
		points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_1)
		points_b = [[sum(pair) for pair in zip(point, translation_2)] for point in original_a]
		points_c , _ = regular_polygon(sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_3)
	
	elif category == 1 and regular and not displace_vertices:
		points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_1)
		points_b , _ = regular_polygon(sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_2)
		points_c = [[sum(pair) for pair in zip(point, translation_3)] for point in original_a]
	
	elif category == 0 and not regular and not displace_vertices:
		points_a , original_a = iregular_polygon_from_regular(
			sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_1, max_dev=max_dev_1)
		points_b = [[sum(pair) for pair in zip(point, translation_2)] for point in original_a]
		points_c , _ = iregular_polygon_from_regular(
			sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_3, max_dev=max_dev_2)
	
	elif category == 1 and not regular and not displace_vertices:
		points_a , original_a = iregular_polygon_from_regular(
			sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_1, max_dev=max_dev_1)
		points_b , _ = iregular_polygon_from_regular(
			sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_2, max_dev=max_dev_2)
		points_c = [[sum(pair) for pair in zip(point, translation_3)] for point in original_a]

	elif category == 0 and regular and displace_vertices:
		# The negative case is the original poligon with parts displaced.
		points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_1)
		points_b = [[sum(pair) for pair in zip(point, translation_2)] for point in original_a]
		points_c = [[sum(pair) for pair in zip(point, translation_3)] for point in original_a]
	
	elif category == 1 and regular and displace_vertices:
		# A math.pi/4 (45 degrees) rotation gives the most stable polygons in the "1" category.
		points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_1)
		points_b = [[sum(pair) for pair in zip(point, translation_2)] for point in original_a]
		points_c = [[sum(pair) for pair in zip(point, translation_3)] for point in original_a]
	
	else:
		raise ValueError('wrong category or regular args!') 
	
	# Get the translation of the 'different' object for the scrambled condition
	translation_diff = translation_3 if category == 0 else translation_2

	return points_a, points_b, points_c, tuple(translation_diff), radius_1

def svrt_1_img(
	category=1,
	radii=None,
	sides=None,
	regular=False,
	rotations=None,
	irregularity=0.5,
	thickness=1,
	color_a=None,
	color_b=None,
	color_c=None,
	filled=False,
	closed=True,
	displace_vertices=False,
	separated_chanels=False):
	"""Returns a picture of single instance of a MTS problem.
	Args:
		category: 0 (lef) or 1 (right).
		radii: radii of the base regular polygon. 2-tuple 8 to 14.
		sides: number of sides of the base regular polygon. 2-tuple 4 to 8.
		rotations: rotations of the polygons. 2-tuple 4 to 8.
		regular: whether to build regular or irregular polygons in radiants. 2-tuple form 0 to pi.
		irregularity: maximum level of random point translation for irregular polygons.
		thickness: line width of the shapes.
		color: line color of the shapes.
		separated_chanels: if True returns two images with one object per image.
	Returns:
		Numpy array."""
	
	# Array of size 128x128 filled with ones as values, to create an image with black color.
	img = np.zeros(shape=(128,128,3),dtype=np.int16)
	img[:] = (255, 255, 255)  # Changing the color of the image

	# Create second canvas for the second chanel.
	if separated_chanels:
		img2 = np.zeros(shape=(128,128,3),dtype=np.int16)
		img2[:] = (255, 255, 255)
		img3 = np.zeros(shape=(128,128,3),dtype=np.int16)
		img3[:] = (255, 255, 255)

	# Set up line color.
	color_a = (0,0,0) if color_a is None else color_a

	# By default make the color of the second object the same as the first one.
	if color_b is None:
		color_b = color_a
	if color_c is None:
		color_c = color_a

	# Get points.
	points_a, points_b, points_c, midpoint_diff, radius_diff = svrt_1_points(
		category=category,
		radii=radii,
		sides=sides,
		rotations=rotations,
		regular=regular,
		irregularity=irregularity,
		displace_vertices=displace_vertices)

	# Assigning sides to polygon
	poly_a = np.array(points_a,dtype=np.int32)
	poly_b = np.array(points_b,dtype=np.int32)
	poly_c = np.array(points_c,dtype=np.int32)

	# Reshaping according to opencv format
	poly_new_a = poly_a.reshape((-1,1,2))
	poly_new_b = poly_b.reshape((-1,1,2))
	poly_new_c = poly_c.reshape((-1,1,2))

	# Draw.
	if not filled and not displace_vertices:
		cv2.polylines(img,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
		if separated_chanels:
			cv2.polylines(img2,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
			cv2.polylines(img3,[poly_new_c],isClosed=closed,color=color_c,thickness=thickness)
		else:
			cv2.polylines(img,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
			cv2.polylines(img,[poly_new_c],isClosed=closed,color=color_c,thickness=thickness)

	elif not filled and displace_vertices and category == 1:
		cv2.polylines(img,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
		if separated_chanels:
			cv2.polylines(img2,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
			img2 = scramble_poligon(img2, midpoint=midpoint_diff, radius=radius_diff)
			cv2.polylines(img3,[poly_new_c],isClosed=closed,color=color_c,thickness=thickness)
		else:
			cv2.polylines(img,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
			cv2.polylines(img,[poly_new_c],isClosed=closed,color=color_c,thickness=thickness)
			img = scramble_poligon(img, midpoint=midpoint_diff, radius=radius_diff)

	elif not filled and displace_vertices and category == 0:
		cv2.polylines(img,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
		if separated_chanels:
			cv2.polylines(img2,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
			cv2.polylines(img3,[poly_new_c],isClosed=closed,color=color_c,thickness=thickness)
			img3 = scramble_poligon(img3, midpoint=midpoint_diff, radius=radius_diff)
		else:
			cv2.polylines(img,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
			cv2.polylines(img,[poly_new_c],isClosed=closed,color=color_c,thickness=thickness)
			img = scramble_poligon(img, midpoint=midpoint_diff, radius=radius_diff)
	
	else:
		cv2.fillPoly(img, [poly_new_a], color=color_a)
		if separated_chanels:
			cv2.fillPoly(img2, [poly_new_b], color=color_b)
			cv2.fillPoly(img3, [poly_new_c], color=color_c)
		else:
			cv2.fillPoly(img, [poly_new_b], color=color_b)
			cv2.fillPoly(img, [poly_new_c], color=color_c)
	
	# Return image(s).
	if separated_chanels:
		return img.astype('uint8'), img2.astype('uint8'), img3.astype('uint8')
	else:
		return img.astype('uint8')


# Arrows
def rotate(origin, point, angle):
	"""Rotate a point counterclockwise by a given angle around a given origin.
	Because in OpenCV the y-axis is inverted this function swaps the x and y axis.
	Args:
		origin: (x, y) tuple.
		point: the point (x, y) to rotate.
		angle: in radiants.

	The angle should be given in radians.
	"""
	oy, ox = origin
	py, px = point

	qx = ox + int(math.cos(angle) * (px - ox)) - int(math.sin(angle) * (py - oy))
	qy = oy + int(math.sin(angle) * (px - ox)) + int(math.cos(angle) * (py - oy))
	return int(qy), int(qx)

def rotate_and_translate(origin, point_list, angle, translation):
	"""Rotate polygon points counterclockwise by a given angle around a given origin and translate.
	Args:
		origin: (x, y) tuple.
		point_list: list of points (x, y) to rotate.
		angle: in degrees.
	Returns:
		New list of points rotated and translated.
	"""
	# Get angle in ratiants.
	radiants = math.radians(angle)

	# Rotate all points.
	new_points = [rotate(origin=origin, point=p, angle=radiants) for p in point_list]

	# Translate all points.
	new_points = [[sum(pair) for pair in zip(point, translation)] for point in new_points]

	return new_points

def get_triangle_top_midpoint(point_list):
	"""Returns the midpoint of the top of a triangle regardless of the orientation."""

	y = int(min([x[1] for x in point_list]))
	x = int((min([x[0] for x in point_list]) + max([x[0] for x in point_list])) / 2)

	return x, y

def get_triangle_bottom_midpoint(point_list):
	"""Returns the midpoint of the top of a triangle regardless of the orientation."""

	y = int(max([x[1] for x in point_list]))
	x = int((min([x[0] for x in point_list]) + max([x[0] for x in point_list])) / 2)

	return x, y

def get_arrow_points(radius, center, rotation=0, shape_a='normal', shape_b='normal', continuous=True):
	"""Calculates the points for a arrow.
	Args:
		radius: of the base circle to build the triangles (heads). 5, 7, 9 works well.
		rotation: of the arrow in degrees.
		center: center of the arrow.
		shape_a: shape of head a. "normal", "inverted".
		shape_b: shape of head b. "normal", "inverted".
		continuous: wether the line touches the available heads.
	Returns:
		3 lists of lists of points. the first is the "top" head, the second the "bottom" and the third is the line.
	"""

	# The base arrow is based on 4 circles.
	# The overall centre is at 2 radi from the top head centre.
	origin_top = (center[0], int(center[1]-2*radius))
	origin_bottom = [center[0], int(center[1]+2*radius)]

	points_top, cero_centered_top = regular_polygon(sides=3,
													radius=radius,
													rotation=math.radians(180),
													translation=origin_top)
	# Use the same function to generate the bottom!
	points_bottom, cero_centered_bottom = regular_polygon(sides=3,
													radius=radius,
													rotation=math.radians(0),
													translation=origin_bottom)

	# Get line points.
	top_mid_point = get_triangle_bottom_midpoint(points_top)
	bottom_mid_point = get_triangle_top_midpoint(points_bottom)

	# If the arrow isn't continious shrink the line.
	if not continuous:
		separation = int(radius)
		top_mid_point = center[0], top_mid_point[1] + separation
		bottom_mid_point = center[0], bottom_mid_point[1] - separation

	points_line = [top_mid_point, bottom_mid_point]

	if shape_a == 'inverted':
		# - radius/2.
		origin_top = [origin_top[0], int(origin_top[1]-radius/2)]
		points_top, cero_centered_top = regular_polygon(sides=3,
													radius=radius,
													rotation=math.radians(0),
													translation=origin_top)

	if shape_b == 'inverted':
		# + radius/2.
		origin_bottom = [origin_bottom[0], int(origin_bottom[1]+radius/2)+1]
		points_bottom, cero_centered_bottom = regular_polygon(sides=3,
													radius=radius,
													rotation=math.radians(180),
													translation=origin_bottom)

	# Get angle in ratiants.
	radiants = math.radians(rotation)

	# Rotate all elements the given amount.
	points_top = [rotate(origin=center, point=p, angle=radiants) for p in points_top]
	points_bottom = [rotate(origin=center, point=p, angle=radiants) for p in points_bottom]
	points_line = [rotate(origin=center, point=p, angle=radiants) for p in points_line]


	return points_top, points_bottom, points_line

def sample_midpoints_arrows(size):
	"""Samples midpoints to arrows if sizes 5, 7 or 9 into a 128x128 image."""

	xs = random.sample(list(range(size*4, 127-size*4)), 2)
	ys = random.sample(list(range(size*4, 127-size*4)), 2)
	point_1 = [xs[0], ys[0]]
	point_2 = [xs[1], ys[1]]

	return point_1, point_2

def sample_midpoints_lines(sizes):
	"""Samples midpoints to arrows if sizes 5, 7 or 9 into a 128x128 image."""

	size_1, size_2 = sizes
	x_1 = random.sample(list(range(int(size_1/2)+2, 127-int(size_1/2+2))), 1)[0]
	y_1 = random.sample(list(range(int(size_1/2)+2, 127-int(size_1/2+2))), 1)[0]
	x_2 = random.sample(list(range(int(size_2/2)+2, 127-int(size_2/2+2))), 1)[0]
	y_2 = random.sample(list(range(int(size_2/2)+2, 127-int(size_2/2+2))), 1)[0]
	point_1 = (x_1, y_1)
	point_2 = (x_2, y_2)

	return point_1, point_2

# Straight lines
def get_line_points(size, rotation, center):
	radius = size/2
	angle_1 = math.radians(rotation)
	angle_2 = math.radians(rotation+180)

	x_1 = int(center[0] + int(radius * math.cos(angle_1)))
	y_1 = int(center[1] + int(radius * math.sin(angle_1)))

	x_2 = int(center[0] + int(radius * math.cos(angle_2)))
	y_2 = int(center[1] + int(radius * math.sin(angle_2)))

	return [(x_1, y_1), (x_2, y_2)]

def make_straingt_lines_sd(category, var_factor, line_thickness=1, separated_chanels=False):
	# Background image.               
	img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
	img[:] = (255, 255, 255)
	if separated_chanels:
		img2 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img2[:] = (255, 255, 255)
		img3 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img3[:] = (255, 255, 255)

	# Sample sizes.
	sizes = random.sample(list(range(16, 64, 4)), 2)
	size_1 = sizes[0]
	size_2 = sizes[1]

	# Sample rotations.
	rotations = random.sample([0, 45, 90, 135], 2)
	rotation_1 = rotations[0]
	rotation_2 = rotations[1]

	# Assign size and rotation based on category and the variation factor.
	size_a = size_1
	rotation_a = rotation_1
	if category==1:
		# Left arrow (different)
		size_b = size_1 if var_factor=='rotation' else size_2
		rotation_b = rotation_1 if var_factor=='size' else rotation_2
		# Right arrow (same)
		size_c = size_1
		rotation_c = rotation_1

	if category==0:
		# Left arrow (same)
		size_b = size_1
		rotation_b = rotation_1
		# Right arrow (different)
		size_c = size_1 if var_factor=='rotation' else size_2
		rotation_c = rotation_1 if var_factor=='size' else rotation_2

	# Positions
	translation_1 = [np.random.randint(58, 70), np.random.randint(26, 38)]
	translation_2 = [np.random.randint(26, 38), np.random.randint(90, 102)]
	translation_3 = [np.random.randint(90, 102), np.random.randint(90, 102)]

	# Get arrow points.
	points_line_a = get_line_points(size=size_a, rotation=rotation_a, center=translation_1)
	points_line_b = get_line_points(size=size_b, rotation=rotation_b, center=translation_2)
	points_line_c = get_line_points(size=size_c, rotation=rotation_c, center=translation_3)

	# Draw!
	cv2.line(img, points_line_a[0], points_line_a[1], (0, 0, 0), thickness=line_thickness)
	if separated_chanels:
		cv2.line(img2, points_line_b[0], points_line_b[1], (0, 0, 0), thickness=line_thickness)
		cv2.line(img3, points_line_c[0], points_line_c[1], (0, 0, 0), thickness=line_thickness)
	else:
		cv2.line(img, points_line_b[0], points_line_b[1], (0, 0, 0), thickness=line_thickness)
		cv2.line(img, points_line_c[0], points_line_c[1], (0, 0, 0), thickness=line_thickness)
	
	if separated_chanels:
		return img.astype('uint8'), img2.astype('uint8'), img3.astype('uint8')
	else:
		return img.astype('uint8')

def make_squares_sd(category, line_thickness=1):
	# Background image.               
	img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
	img[:] = (255, 255, 255)

	# Sample sizes.
	sizes = random.sample(list(range(16, 62, 2)), 2)
	size_1 = sizes[0]
	size_2 = sizes[1]

	# Assign size_2 based on category.
	if category==1:
		size_2 = size_1

	# Sample start and end points.
	x_1 = random.sample(list(range(2, 127-(size_1+2))), 1)[0]
	y_1 = random.sample(list(range(2, 127-(size_1+2))), 1)[0]
	x_2 = random.sample(list(range(2, 127-(size_2+2))), 1)[0]
	y_2 = random.sample(list(range(2, 127-(size_2+2))), 1)[0]
	start_point_1 = (x_1, y_1)
	start_point_2 = (x_2, y_2)
	end_point_1 = (x_1+size_1, y_1+size_1)
	end_point_2 = (x_2+size_2, y_2+size_2)

	# Draw squares.
	img = cv2.rectangle(img, start_point_1, end_point_1, (0, 0, 0), 1)
	img = cv2.rectangle(img, start_point_2, end_point_2, (0, 0, 0), 1)

	return img.astype('uint8')

def make_rectangles_sd(category, line_thickness=1, separated_chanels=False):
	# Background image.               
	img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
	img[:] = (255, 255, 255)
	if separated_chanels:
		img2 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img2[:] = (255, 255, 255)
		img3 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img3[:] = (255, 255, 255)

	# Sample constant dimension.
	const_dim = 'x' if random.random() > 0.5 else 'y'

	# Sample sizes.
	if const_dim == 'y':
		sizes_x = random.sample(list(range(16, 64, 4)), 2)
		size_x_1 = sizes_x[0]
		size_x_2 = sizes_x[1]

		size_y_1 = random.sample(list(range(16, 64, 4)), 1)[0]
		size_y_2 = size_y_1
	
	elif const_dim == 'x':
		sizes_y = random.sample(list(range(16, 64, 4)), 2)
		size_y_1 = sizes_y[0]
		size_y_2 = sizes_y[1]

		size_x_1 = random.sample(list(range(16, 64, 4)), 1)[0]
		size_x_2 = size_x_1

	# Assign sizes to the 3 objects
	size_x_a = size_x_1
	size_y_a = size_y_1
	if category==0:
		size_x_b = size_x_1
		size_y_b = size_y_1
		size_x_c = size_x_2
		size_y_c = size_y_2
	elif category==1:
		size_x_b = size_x_2
		size_y_b = size_y_2
		size_x_c = size_x_1
		size_y_c = size_y_1

	# Sample start and end points.
	translation_1 = [np.random.randint(58, 70), np.random.randint(26, 38)]
	translation_2 = [np.random.randint(26, 38), np.random.randint(90, 102)]
	translation_3 = [np.random.randint(90, 102), np.random.randint(90, 102)]
	
	start_point_a = (translation_1[0] - int(size_x_a/2), translation_1[1] - int(size_y_a/2))
	start_point_b = (translation_2[0] - int(size_x_b/2), translation_2[1] - int(size_y_b/2))
	start_point_c = (translation_3[0] - int(size_x_c/2), translation_3[1] - int(size_y_c/2))
	end_point_a = (start_point_a[0] + size_x_a, start_point_a[1] + size_y_a)
	end_point_b = (start_point_b[0] + size_x_b, start_point_b[1] + size_y_b)
	end_point_c = (start_point_c[0] + size_x_c, start_point_c[1] + size_y_c)

	# Draw squares.
	img = cv2.rectangle(img, start_point_a, end_point_a, (0, 0, 0), 1)
	if separated_chanels:
		img2 = cv2.rectangle(img2, start_point_b, end_point_b, (0, 0, 0), 1)
		img3 = cv2.rectangle(img3, start_point_c, end_point_c, (0, 0, 0), 1)
	else:
		img = cv2.rectangle(img, start_point_b, end_point_b, (0, 0, 0), 1)
		img = cv2.rectangle(img, start_point_c, end_point_c, (0, 0, 0), 1)

	if separated_chanels:
		return img.astype('uint8'), img2.astype('uint8'), img3.astype('uint8')
	else:
		return img.astype('uint8')

def make_connected_open_squares(category, line_width=1, is_closed=False, separated_chanels=False):
	# Background image.               
	img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
	img[:] = (255, 255, 255)
	if separated_chanels:
		img2 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img2[:] = (255, 255, 255)
		img3 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img3[:] = (255, 255, 255)

	# Sample segment size.
	size = random.randint(10, 24)

	# Define figure points.
	points_a = [
		[0, size],
		[0, 0],
		[size, 0],
		[size, size],
		[size, 2*size],
		[2*size, 2*size],
		[2*size, size]
		]

	points_diff = [
		[0, size],
		[0, 2*size],
		[size, 2*size],
		[size, size],
		[size, 0],
		[2*size, 0],
		[2*size, size]
		]
	# Assign points based on category.
	if category == 0:
		points_b = points_a
		points_c = points_diff
	elif category == 1:
		points_b = points_diff
		points_c = points_a
	else:
		raise ValueError('category can only be 0 (left) or 1 (right)!')

	# Sample translations and apply.
	translation_1 = [np.random.randint(58-size, 70-size), np.random.randint(26-size, 38-size)]
	translation_2 = [np.random.randint(26-size, 38-size), np.random.randint(90-size, 102-size)]
	translation_3 = [np.random.randint(90-size, 102-size), np.random.randint(90-size, 102-size)]

	points_a = [[sum(pair) for pair in zip(point, translation_1)] for point in points_a]
	points_b = [[sum(pair) for pair in zip(point, translation_2)] for point in points_b]
	points_c = [[sum(pair) for pair in zip(point, translation_3)] for point in points_c]

	# Assigning sides to polygon
	poly_a = np.array(points_a,dtype=np.int32)
	poly_b = np.array(points_b,dtype=np.int32)
	poly_c = np.array(points_c,dtype=np.int32)

	# Reshaping according to opencv format
	poly_new_a = poly_a.reshape((-1,1,2))
	poly_new_b = poly_b.reshape((-1,1,2))
	poly_new_c = poly_c.reshape((-1,1,2))

	# Draw.
	cv2.polylines(img,[poly_new_a],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
	if separated_chanels:
		cv2.polylines(img2,[poly_new_b],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
		cv2.polylines(img3,[poly_new_c],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
	else:
		cv2.polylines(img,[poly_new_b],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
		cv2.polylines(img,[poly_new_c],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)

	if separated_chanels:
		return img, img2
	else:
		return img

def make_connected_circles(category, line_thickness=1, separated_chanels=False):
	# Background image.               
	img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
	img[:] = (255, 255, 255)
	if separated_chanels:
		img2 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img2[:] = (255, 255, 255)
		img3 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img3[:] = (255, 255, 255)
	
	# Get small and big radious.
	radii = random.sample(list(range(6, 20, 4)), 2)
	radii.sort()
	radius_small = radii[0]
	radius_big = radii[1]
	
	# Sample ordering of circes: small-big vs big-small.
	order_1 = 'sb' if random.random() < 0.5 else 'bs'
	order_2 = 'bs' if order_1 == 'sb' else 'sb'
	
	# Assign radii based on order.
	radius_1_a = radius_small if order_1 == 'sb' else radius_big
	radius_1_b = radius_big if order_1 == 'sb' else radius_small
	radius_2_1 = radius_small if order_2 == 'sb' else radius_big
	radius_2_2 = radius_big if order_2 == 'sb' else radius_small

	# Assign radius_big based on category.
	# if category==1:
	# 		radius_2_a = radius_1_a
	# 		radius_2_b = radius_1_b
	
	if category==0:
		radius_2_a = radius_1_a
		radius_2_b = radius_1_b
		radius_3_a = radius_2_1
		radius_3_b = radius_2_2
	elif category==1:
		radius_2_a = radius_2_1
		radius_2_b = radius_2_2
		radius_3_a = radius_1_a
		radius_3_b = radius_1_b

	# Sample midpoints.
	# mpdt_1 = (
	# 	np.random.randint(radius_big+2, 126-radius_big),
	# 	np.random.randint(radius_1_a+radius_1_b+2, 126-(radius_1_a+radius_1_b))
	# 	)
	
	# mpdt_2 = (
	# 	np.random.randint(radius_big+2, 127-radius_big),
	# 	np.random.randint(radius_2_a+radius_2_b+2, 126-(radius_2_a+radius_2_b))
	# 	)

	# Sample start and end points.
	mpdt_1 = [np.random.randint(58, 70), np.random.randint(26, 38)]
	mpdt_2 = [np.random.randint(26, 38), np.random.randint(90, 102)]
	mpdt_3 = [np.random.randint(90, 102), np.random.randint(90, 102)]

	mdpt_1_a = (mpdt_1[0], mpdt_1[1]-radius_1_b)
	mdpt_1_b = (mpdt_1[0], mpdt_1[1]+radius_1_a)

	mdpt_2_a = (mpdt_2[0], mpdt_2[1]-radius_2_b)
	mdpt_2_b = (mpdt_2[0], mpdt_2[1]+radius_2_a)

	mdpt_3_a = (mpdt_3[0], mpdt_3[1]-radius_3_b)
	mdpt_3_b = (mpdt_3[0], mpdt_3[1]+radius_3_a)


	# Draw circles.
	img = cv2.circle(img, mdpt_1_a, radius_1_a, (0, 0, 0), 1)
	img = cv2.circle(img, mdpt_1_b, radius_1_b, (0, 0, 0), 1)
	if separated_chanels:
		img2 = cv2.circle(img2, mdpt_2_a, radius_2_a, (0, 0, 0), 1)
		img2 = cv2.circle(img2, mdpt_2_b, radius_2_b, (0, 0, 0), 1)
		img3 = cv2.circle(img3, mdpt_3_a, radius_3_a, (0, 0, 0), 1)
		img3 = cv2.circle(img3, mdpt_3_b, radius_3_b, (0, 0, 0), 1)
	else:
		img = cv2.circle(img, mdpt_2_a, radius_2_a, (0, 0, 0), 1)
		img = cv2.circle(img, mdpt_2_b, radius_2_b, (0, 0, 0), 1)
		img = cv2.circle(img, mdpt_3_a, radius_3_a, (0, 0, 0), 1)
		img = cv2.circle(img, mdpt_3_b, radius_3_b, (0, 0, 0), 1)
	
	if separated_chanels:
		return img, img2, img3
	else:
		return img

def make_arrows_sd(
	category,
	continuous=True,
	line_width=1,
	separated_chanels=False):
	"""
	Args:
		category: 1 (left) or 0 (right).
		continuous: weather the line touches the available heads.
		line_width: line width in pixels.
	Returns:
		image (array).
	"""
	
	# Background image.               
	img = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
	img[:] = (255, 255, 255)
	if separated_chanels:
		img2 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img2[:] = (255, 255, 255)
		img3 = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
		img3[:] = (255, 255, 255)

	# Sample sizes.
	size_1 = random.sample([4, 5, 6, 7, 8, 9, 10], 1)[0]
	size_2 = random.sample([4, 5, 6, 7, 8, 9, 10], 1)[0]
	
	# Sample rotations.
	rotation_1 = random.randint(0, 135)
	rotation_2 = random.randint(0, 135)

	# Sample booleans for arrows:
	head_a_1 = bool(random.sample([0, 1], 1)[0])
	head_b_1 = True if not head_a_1 else bool(random.sample([0, 1], 1)[0])
	head_a_2 = bool(random.sample([0, 1], 1)[0])
	head_b_2 = True if not head_a_2 else bool(random.sample([0, 1], 1)[0])

	# Sample shapes.
	shape_a_1 = random.sample(['normal', 'inverted'], 1)[0]
	shape_b_1 = random.sample(['normal', 'inverted'], 1)[0]
	shape_a_2 = random.sample(['normal', 'inverted'], 1)[0]
	shape_b_2 = random.sample(['normal', 'inverted'], 1)[0]

	# Assign size_b and rotation_b based on category.
	if category==1:
		# Left arrow
		size_2 = size_1
		rotation_2 = rotation_1
		# Ensure that the values of shape_a_1 and shape_b_1 are different.
		shape_a_1 = 'inverted' if shape_b_1 == 'inverted' else 'normal'
		# Ensure that the second arrow has the oposite head orientations.
		shape_a_2 = 'inverted' if shape_b_1 == 'normal' else 'normal'
		shape_b_2 = 'inverted' if  shape_a_1 == 'normal' else 'normal'
		# Set up arrow heads (each arrow has at least one head)
		if not head_a_1:
			head_b_2 = False
			head_a_2 = True
		if not head_b_1:
			head_a_2 = False
			head_b_2 = True
		if head_a_1 and head_b_1:
			head_a_2 = head_a_1
			head_b_2 = head_b_1

		# Right arrow
		size_3 = size_1
		rotation_3 = rotation_1
		head_a_3 = head_a_1
		head_b_3 = head_b_1
		shape_a_3 = shape_a_1
		shape_b_3 = shape_b_1
	
	if category==0:
		# Left arrow
		size_2 = size_1
		rotation_2 = rotation_1
		head_a_2 = head_a_1
		head_b_2 = head_b_1
		shape_a_2 = shape_a_1
		shape_b_2 = shape_b_1

		# Right arrow
		size_3 = size_1
		rotation_3 = rotation_1
		# Ensure that the values of shape_a_1 and shape_b_1 are different.
		shape_a_3 = 'inverted' if shape_b_1 == 'inverted' else 'normal'
		# Ensure that the second arrow has the oposite head orientations.
		shape_a_3 = 'inverted' if shape_b_1 == 'normal' else 'normal'
		shape_b_3 = 'inverted' if  shape_a_1 == 'normal' else 'normal'
		# Set up arrow heads (each arrow has at least one head)
		if not head_a_1:
			head_b_3 = False
			head_a_3 = True
		if not head_b_1:
			head_a_3 = False
			head_b_3 = True
		if head_a_1 and head_b_1:
			head_a_3 = head_a_1
			head_b_3 = head_b_1

	# Get midpoints.
	# midpoint_1, midpoint_2 = sample_midpoints_arrows(size=size_1)

	# Positions
	translation_1 = [np.random.randint(58, 70), np.random.randint(26, 38)]
	translation_2 = [np.random.randint(26, 38), np.random.randint(90, 102)]
	translation_3 = [np.random.randint(90, 102), np.random.randint(90, 102)]

	# Get arrow points.
	points_top_1, points_bottom_1, points_line_1 = get_arrow_points(radius=size_1,
																	rotation=rotation_1,
																	shape_a=shape_a_1, shape_b=shape_b_1,
																	center=translation_1,
																	continuous=continuous)

	points_top_2, points_bottom_2, points_line_2 = get_arrow_points(radius=size_2,
																	rotation=rotation_2,
																	shape_a=shape_a_2, shape_b=shape_b_2,
																	center=translation_2,
																	continuous=continuous)

	points_top_3, points_bottom_3, points_line_3 = get_arrow_points(radius=size_3,
																	rotation=rotation_3,
																	shape_a=shape_a_3, shape_b=shape_b_3,
																	center=translation_3,
																	continuous=continuous)

	# Reshape arrow points according to opencv format.
	poly_top_1 = np.array(points_top_1, dtype=np.int32)
	poly_bottom_1 = np.array(points_bottom_1, dtype=np.int32)
	poly_new_bottom_1 = poly_bottom_1.reshape((-1,1,2))
	poly_new_top_1 = poly_top_1.reshape((-1,1,2))

	poly_top_2 = np.array(points_top_2, dtype=np.int32)
	poly_bottom_2 = np.array(points_bottom_2, dtype=np.int32)
	poly_new_bottom_2 = poly_bottom_2.reshape((-1,1,2))
	poly_new_top_2 = poly_top_2.reshape((-1,1,2))

	poly_top_3 = np.array(points_top_3, dtype=np.int32)
	poly_bottom_3 = np.array(points_bottom_3, dtype=np.int32)
	poly_new_bottom_3 = poly_bottom_3.reshape((-1,1,2))
	poly_new_top_3 = poly_top_3.reshape((-1,1,2))

	# Draw!
	if head_a_1:
		cv2.polylines(img,[poly_new_top_1],isClosed=True,color=(0, 0, 0),thickness=line_width)
	if head_b_1:
		cv2.polylines(img,[poly_new_bottom_1],isClosed=True,color=(0, 0, 0),thickness=line_width)
	cv2.line(img, points_line_1[0], points_line_1[1], (0, 0, 0), thickness=line_width)

	if separated_chanels:
		if head_a_2:
			cv2.polylines(img2,[poly_new_top_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
		if head_b_2:
			cv2.polylines(img2,[poly_new_bottom_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
		cv2.line(img2, points_line_2[0], points_line_2[1], (0, 0, 0), thickness=line_width)

		if head_a_3:
			cv2.polylines(img3,[poly_new_top_3],isClosed=True,color=(0, 0, 0),thickness=line_width)
		if head_b_3:
			cv2.polylines(img3,[poly_new_bottom_3],isClosed=True,color=(0, 0, 0),thickness=line_width)
		cv2.line(img3, points_line_3[0], points_line_3[1], (0, 0, 0), thickness=line_width)

	else:
		if head_a_2:
			cv2.polylines(img,[poly_new_top_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
		if head_b_2:
			cv2.polylines(img,[poly_new_bottom_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
		cv2.line(img, points_line_2[0], points_line_2[1], (0, 0, 0), thickness=line_width)

		if head_a_3:
			cv2.polylines(img,[poly_new_top_3],isClosed=True,color=(0, 0, 0),thickness=line_width)
		if head_b_3:
			cv2.polylines(img,[poly_new_bottom_3],isClosed=True,color=(0, 0, 0),thickness=line_width)
		cv2.line(img, points_line_3[0], points_line_3[1], (0, 0, 0), thickness=line_width)
	
	if separated_chanels:
		return img.astype('uint8'), img2.astype('uint8'), img3.astype('uint8')
	else:
		return img.astype('uint8')

# Define generators single chanel.
def problem_1_irregular_polygon_gen(batch_size=64, category_type='both'):
	"""Generatot for problem 1, test 1 (irregular polygon).
	Test 1: shape=irregular, color=black, thickness=1, inverted=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		sd_labels = []
		coordinates = []
		rel_positions = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=False, color_a=(0,0,0), sides=None, thickness=1)
			coors, rel_pos, bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				sd_labels.append(y)
				coordinates.append(coors)
				rel_positions.append(rel_pos)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), [np.array(sd_labels), np.array(coordinates), np.array(rel_positions)]

def problem_1_regular_polygon_gen(batch_size=64, category_type='both'):
	"""Generatot for problem 1, test 2 (regular polygon).
	Test 2: shape=regular, color=black, thickness=1, inverted=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		sd_labels = []
		coordinates = []
		rel_positions = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=True, color_a=(0,0,0), sides=None, thickness=1)
			coors, rel_pos, bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				sd_labels.append(y)
				coordinates.append(coors)
				rel_positions.append(rel_pos)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), [np.array(sd_labels), np.array(coordinates), np.array(rel_positions)]



def get_shapes_info(img, scrambled_negative=False, draw_b_rects=False):
	"""Checks whether objects are touching each others or the image boundaries.
	Args:
		img: image (np.array)
		scrambled_negative: whether img is a negative example from the scrambled condition.
	Returns:
		flag: True if the objects are touching each other or the image boundaries. Used to discard the sample."""

	## Get image ready
	image = img.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

	# Find contours, obtain contours
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1] # this is to deal with different opencv versions.

	# Get bounding boxes
	b_rects = []
	for c in cnts:
		b_rects.append(cv2.boundingRect(c))
	if scrambled_negative:
		# Get the two bigest bounding box.
		new_b_rects = []
		big_rects = sorted(b_rects,key=lambda x: x[2]*x[3], reverse=True)[:2]
		new_b_rects.extend(big_rects)
		# Get all smaller bounding boxes.
		smaller_b_rects = [x for x in b_rects if x not in new_b_rects]
	
		# Build second bounding box enclosing all smaller bounding boxes.
		if len(smaller_b_rects) > 0:
			min_x = min(enumerate(smaller_b_rects),key=lambda x: x[1][0])[1][0]
			min_y = min(enumerate(smaller_b_rects),key=lambda x: x[1][1])[1][1]
			max_x = max(enumerate(smaller_b_rects),key=lambda x: x[1][0]+x[1][2])[1]
			max_y = max(enumerate(smaller_b_rects),key=lambda x: x[1][1]+x[1][3])[1]
			max_x = max_x[0] + max_x[2]
			max_y = max_y[1] + max_y[3]
			w = max_x - min_x
			h = max_y - min_y
			new_b_rect = (min_x, min_y, w, h)
			new_b_rects.append(new_b_rect)
			b_rects = new_b_rects
		# If there are none smaller rects, set b_rects to an empty list to activate flag
		else:
			b_rects = []
	
	# Get flag value
	# There has to be exactly 3 objects
	flag = False if len(b_rects) == 3 else True

	if not flag:
		height, width = thresh.shape[:2]
		touching_edge = []
		for rect in b_rects:
			# get bounds
			x, y, w, h = rect
			# check if touching edge
			on_edge = False
			if x <= 0 or (x + w) >= (width - 1):
				on_edge = True
			if y <= 0 or (y + h) >= (height - 1):
				on_edge = True
			# add to list
			touching_edge.append(on_edge)
		if any(touching_edge):
			flag = True
	
	if draw_b_rects:
		# Draw bounding rects.
		for b in b_rects:
			x,y,w,h = b
			cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
		# Show image
		# plt.imshow(img)
		# plt.show()
	
	return flag



# Define generators single chanel.
def problem_1_irregular_polygon_gen(batch_size=64, category_type='both'):
	"""Generatot for problem 1, test 1 (irregular polygon).
	Test 1: shape=irregular, color=black, thickness=1, inverted=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=False, color_a=(0,0,0), sides=None, thickness=1)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_regular_polygon_gen(batch_size=64, category_type='both'):
	"""Generatot for problem 1, test 2 (regular polygon).
	Test 2: shape=regular, color=black, thickness=1, inverted=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=True, color_a=(0,0,0), sides=None, thickness=1)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_open_gen(closed=False, batch_size=64, category_type='both'):
	"""Generator for problem 1, test 8 (open).
	Test 1: shape=irregular, color=black, thickness=1, inverted=not, closed=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")

		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=False, color_a=(0,0,0), sides=None,
						thickness=1, closed=closed)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_wider_line_gen(thickness=2, batch_size=64, category_type='both'):
	"""Generatot for problem 1, test 4 (wider line).
	Test 4: shape=irregular, color=black, thickness=[2, 3, 4, 5], inverted=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=False, color_a=(0,0,0), sides=None, thickness=thickness)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_scrambled_gen(batch_size=64, category_type='both'):
	"""Generator for problem 1, test 9 (Scrambled).
	Test 1: shape=rregular, color=black, thickness=1, inverted=not, displace_vertices=True."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)
	
	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(
				category=y, 
				regular=True, 
				color_a=(0,0,0), 
				sides=None, 
				thickness=1, 
				displace_vertices=True
				)
			bad_sample = get_shapes_info(x, scrambled_negative=True, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_random_color_gen(batch_size=64, category_type='both'):
	"""Generatot for problem 1, test 3.
	Test 3: shape=irregular, color=random, thickness=1, inverted=not."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			color = tuple(np.random.randint(1, high=256, size=3))
			color = (int(color[0]), int(color[1]), int(color[2]))
			y = labels[i]
			x = svrt_1_img(category=y, regular=False, color_a=color, sides=None, thickness=1)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_filled_gen(batch_size=64, category_type='both'):
	"""Generator for problem 1, test 6 (filled).
	Test 1: shape=irregular, color=black, thickness=1, inverted=not, filled=True."""
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")

		i = 0
		while True:
			y = labels[i]
			x = svrt_1_img(category=y, regular=False, color_a=(0,0,0), sides=None, thickness=1, filled=True)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def problem_1_arrows_gen(batch_size=64, category_type='both', continuous=True):
	# Check that batch_size even if both categories are being generated.
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = make_arrows_sd(y, continuous=continuous, line_width=1)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def straingt_lines_gen(batch_size=64, category_type='both', var_factor='size'):
    # Check that batch_size even if both categories are being generated.
    if category_type == 'both':
        try:
            assert batch_size % 2 == 0
        except:
            print("batch_size should be an even number!")
        else:
            half_batch = int(batch_size/2)

    while True:
        inputs = []
        labels = []
        if category_type == 'both':
            labels = [0] * half_batch + [1] * half_batch
            random.shuffle(labels)
        elif category_type == '0':
            labels = [0] * batch_size
        elif category_type == '1':
            labels = [1] * batch_size
        else:
            raise ValueError("category_type should be 'both', '0', or '1'!")
        
        i = 0
        while True:
            y = labels[i]
            x = make_straingt_lines_sd(y, var_factor=var_factor, line_thickness=1)
            bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
            if bad_sample:
                pass
            else:
                inputs.append(x)
                labels.append(y)
                i += 1
            if i == batch_size:
                break
        yield np.array(inputs), np.array(labels)

def connected_open_squares_gen(batch_size=64, category_type='both'):
    # Check that batch_size even if both categories are being generated.
    if category_type == 'both':
        try:
            assert batch_size % 2 == 0
        except:
            print("batch_size should be an even number!")
        else:
            half_batch = int(batch_size/2)

    while True:
        inputs = []
        sd_labels = []
        coordinates = []
        rel_positions = []
        if category_type == 'both':
            labels = [0] * half_batch + [1] * half_batch
            random.shuffle(labels)
        elif category_type == '0':
            labels = [0] * batch_size
        elif category_type == '1':
            labels = [1] * batch_size
        else:
            raise ValueError("category_type should be 'both', '0', or '1'!")
        
        i = 0
        while True:
            y = labels[i]
            x = make_connected_open_squares(category=y, line_width=1)
            bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
            if bad_sample:
                pass
            else:
                inputs.append(x)
                labels.append(y)
                i += 1
            if i == batch_size:
                break
        yield np.array(inputs), np.array(labels)

def connected_closed_squares_gen(batch_size=64, category_type='both'):
    # Check that batch_size even if both categories are being generated.
    if category_type == 'both':
        try:
            assert batch_size % 2 == 0
        except:
            print("batch_size should be an even number!")
        else:
            half_batch = int(batch_size/2)

    while True:
        inputs = []
        sd_labels = []
        coordinates = []
        rel_positions = []
        if category_type == 'both':
            labels = [0] * half_batch + [1] * half_batch
            random.shuffle(labels)
        elif category_type == '0':
            labels = [0] * batch_size
        elif category_type == '1':
            labels = [1] * batch_size
        else:
            raise ValueError("category_type should be 'both', '0', or '1'!")
        
        i = 0
        while True:
            y = labels[i]
            x = make_connected_open_squares(category=y, line_width=1, is_closed=True)
            bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
            if bad_sample:
                pass
            else:
                inputs.append(x)
                labels.append(y)
                i += 1
            if i == batch_size:
                break
        yield np.array(inputs), np.array(labels)

def rectangles_gen(batch_size=64, category_type='both'):
	# Check that batch_size even if both categories are being generated.
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = make_rectangles_sd(category=y, line_thickness=1)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)

def connected_circles_gen(batch_size=64, category_type='both'):
	# Check that batch_size even if both categories are being generated.
	if category_type == 'both':
		try:
			assert batch_size % 2 == 0
		except:
			print("batch_size should be an even number!")
		else:
			half_batch = int(batch_size/2)

	while True:
		inputs = []
		labels = []
		if category_type == 'both':
			labels = [0] * half_batch + [1] * half_batch
			random.shuffle(labels)
		elif category_type == '0':
			labels = [0] * batch_size
		elif category_type == '1':
			labels = [1] * batch_size
		else:
			raise ValueError("category_type should be 'both', '0', or '1'!")
		
		i = 0
		while True:
			y = labels[i]
			x = make_connected_circles(category=y, line_thickness=1)
			bad_sample = get_shapes_info(x, scrambled_negative=False, draw_b_rects=False)
			if bad_sample:
				pass
			else:
				inputs.append(x)
				labels.append(y)
				i += 1
			if i == batch_size:
				break
		yield np.array(inputs), np.array(labels)


def check_path(path):
	"""Function for creating directory if it doesn't exist yet"""
	if not os.path.exists(path):
		os.mkdir(path)

def make_datasets(batch_size, train_size, val_size, test_size):
	# Get all generators
	irregular_gen = problem_1_irregular_polygon_gen(batch_size=batch_size, category_type='both')
	regular_gen = problem_1_regular_polygon_gen(batch_size=batch_size, category_type='both')
	open_gen = problem_1_open_gen(batch_size=batch_size, category_type='both')
	wider_gen = problem_1_wider_line_gen(batch_size=batch_size, category_type='both')
	scrambled_gen = problem_1_scrambled_gen(batch_size=batch_size, category_type='both')
	random_gen = problem_1_random_color_gen(batch_size=batch_size, category_type='both')
	filled_gen = problem_1_filled_gen(batch_size=batch_size, category_type='both')
	lines_gen = connected_open_squares_gen(batch_size=batch_size, category_type='both')
	arrows_gen = problem_1_arrows_gen(batch_size=batch_size, category_type='both')
	rect_gen = rectangles_gen(batch_size=batch_size, category_type='both')
	slines_gen = straingt_lines_gen(batch_size=batch_size, var_factor='size', category_type='both')
	csquares_gen = connected_closed_squares_gen(batch_size=batch_size, category_type='both')
	ccircles_gen = connected_circles_gen(batch_size=batch_size, category_type='both')
	datasets = [
		irregular_gen,
		regular_gen,
		open_gen,
		wider_gen,
		scrambled_gen,
		random_gen,
		filled_gen,
		lines_gen,
		arrows_gen,
		rect_gen,
		slines_gen,
		csquares_gen,
		ccircles_gen
		]
	# Get all ds names
	ds_names = [
		'irregular_mts',
		'regular_mts',
		'open_mts', 
		'wider_mts', 
		'scrambled_mts',
		'random_mts',
		'filled_mts',
		'lines_mts',
		'arrows_mts',
		'rectangles_mts',
		'slines_mts',
		'csquares_mts',
		'ccircles_mts'
		]
	splits = ['train', 'val', 'test']
	sizes = [train_size, val_size, test_size]
	# Iterate over ds and ds_names
	for ds, ds_name in zip(datasets, ds_names):
		# Make ds directory
		ds_dir = f'./data/{ds_name}'
		check_path(ds_dir)
		# Get ds generator
		gen = ds
		# Iterate over splits 
		for split, size in zip(splits, sizes):
			ds_dir = f'./data/{ds_name}/{split}'
			check_path(ds_dir)
			# Initialize csv file for image data
			ds_file = f"./data/{ds_name}/{split}_annotations.csv"
			ds_file_header = ['ID', 'label']
			# Generate data
			counter = 0
			# Open the file in write mode
			with open(ds_file, 'w') as f:
				# Create the csv writer
				writer = csv.writer(f)
				# Write header to the csv file
				writer.writerow(ds_file_header)
				for i in range(size):
					# Generate data
					xs, ys = next(gen)
					for j in range(xs.shape[0]):
						# Get data
						img = xs[j]
						label = ys[j]
						row = [f'{counter}.png', label]
						# Save data
						img = Image.fromarray(img)
						img.save(f'{ds_dir}/{counter}.png')
						writer.writerow(row)
						counter += 1
	return

if __name__ == '__main__':
	# Parameters
	BATCH_SIZE = 100
	TRAIN_SIZE = 28000 // BATCH_SIZE
	VAL_SIZE = 5600 // BATCH_SIZE
	TEST_SIZE = 11200 // BATCH_SIZE
	
	# Generate datasets
	make_datasets(
		batch_size=BATCH_SIZE, 
		train_size=TRAIN_SIZE, 
		val_size=VAL_SIZE, 
		test_size=TEST_SIZE
		)
	print('Done!')
