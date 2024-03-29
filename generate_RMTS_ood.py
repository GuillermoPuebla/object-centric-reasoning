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

def open_rectangle(radius=8, x_offset=None, rotation=None, translation=None):
    if rotation is None:
        rotation =  1 * math.pi * np.random.random_sample()

    if x_offset is None:
        x_offset = np.random.randint(8)

    sides = 4
    one_segment = math.pi * 2 / sides
    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    line_1 = points[0:2]
    line_2 = points[2:4]
    line_2 = [[p[0] - x_offset, p[1]] for p in line_2]
    original_lines = copy.copy([line_1, line_2])

    if translation:
        line_1 = [[sum(pair) for pair in zip(point, translation)]
                  for point in line_1]
        line_2 = [[sum(pair) for pair in zip(point, translation)]
                  for point in line_2]
    lines = [line_1, line_2]

    return lines, original_lines

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

def scramble_poligon(img, midpoint, radius, off_x=None, off_y=None):
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
    if not off_x:
        off_x = random.sample(list(range(0, int(radius/2))), 4)
    if not off_y:
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
    """Returns polygon points for a single instance of a SVRT problem 1.
    Args:
        category: 0 (no) or 1 (yes).
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
            radius_1 = 10 # np.random.randint(10, 18)
            radius_2 = radius_1 #if category==1 else np.random.randint(10, 14)
        else:
            radius_1 = np.random.randint(10, 17) # np.random.randint(10, 40)
            radius_2 = np.random.randint(10, 17) #radius_1 #if category==1 else np.random.randint(10, 40)
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
    # As I have a sample rejection step on the generators I can just sample both positions randomly here.
    if displace_vertices:
        t_a_1 = [np.random.randint(min_dev_1, 32-min_dev_1), np.random.randint(min_dev_1, 32-min_dev_1)]
        t_a_2 = [np.random.randint(32+min_dev_1, 64-min_dev_1), np.random.randint(min_dev_1, 32-min_dev_1)]
        t_a_3 = [np.random.randint(min_dev_1, 32-min_dev_1), np.random.randint(32+min_dev_1, 64-min_dev_1)]
        t_a_4 = [np.random.randint(32+min_dev_1, 64-min_dev_1), np.random.randint(32+min_dev_1, 64-min_dev_1)]
        translation_a = random.choice([t_a_1, t_a_2, t_a_3, t_a_4])
    else:
        translation_a = [np.random.randint(min_dev_1, 64-min_dev_1), np.random.randint(min_dev_1, 64-min_dev_1)]
    
    # Ensure that the second shape is at the other side of at least one dimension. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if displace_vertices:    
        dim = random.choice(['x', 'y']) # Chose dimension to send the object to the other side.
        if dim == 'x':
            t2_x = np.random.randint(15, 30) if translation_a[0] > 32 else np.random.randint(35, 50)
        else:
            t2_x = np.random.randint(15, 50)
        if dim == 'y':
            t2_y = np.random.randint(15, 30) if translation_a[1] > 32 else np.random.randint(35, 50)
        else:
            t2_y = np.random.randint(15, 50)
        translation_b = [t2_x, t2_y]
    else:
        translation_b = [np.random.randint(min_dev_2, 64-min_dev_2), np.random.randint(min_dev_2, 64-min_dev_2)]


    # Generate points.
    if category == 0 and regular and not displace_vertices:
        # A math.pi/4 (45 degrees) rotation gives the most stable polygons in the "1" category.
        points_a , _ = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a)
        points_b , _ = regular_polygon(sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_b)
    
    elif category == 1 and regular and not displace_vertices:
        points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a)
        points_b = [[sum(pair) for pair in zip(point, translation_b)] for point in original_a]
    
    elif category == 0 and not regular and not displace_vertices:
        points_a , _ = iregular_polygon_from_regular(
            sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a, max_dev=max_dev_1)
        points_b , _ = iregular_polygon_from_regular(
            sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_b, max_dev=max_dev_2)
    
    elif category == 1 and not regular and not displace_vertices:
        points_a , original_a = iregular_polygon_from_regular(
            sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a, max_dev=max_dev_1)
        points_b = [[sum(pair) for pair in zip(point, translation_b)] for point in original_a]

    elif category == 1 and regular and displace_vertices:
        # A math.pi/4 (45 degrees) rotation gives the most stable polygons in the "1" category.
        points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a)
        points_b = [[sum(pair) for pair in zip(point, translation_b)] for point in original_a]
    
    elif category == 0 and regular and displace_vertices:
        # The negative case is the original poligon with parts displaced.
        points_a , original_a = regular_polygon(sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a)
        points_b = [[sum(pair) for pair in zip(point, translation_b)] for point in original_a]
        # points_b = displace_polygon_vertices(original_a, radius_1)  # this is a list of list of points
        # new_points_b = []
        # for point_list in points_b:
        #     b = [[sum(pair) for pair in zip(point, translation_b)] for point in point_list]
        #     new_points_b.append(b)
        # points_b = new_points_b

    else:
        raise ValueError('wrong category or regular args!') 
    
    return points_a, points_b, tuple(translation_b), radius_1

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
    filled=False,
    closed=True,
    displace_vertices=False
    ):
    """Returns a picture of single instance of a SVRT problem 1.
    Args:
        category: 0 (no) or 1 (yes).
        radii: radii of the base regular polygon. 2-tuple 8 to 14.
        sides: number of sides of the base regular polygon. 2-tuple 4 to 8.
        rotations: rotations of the polygons. 2-tuple 4 to 8.
        regular: whether to build regular or irregular polygons in radiants. 2-tuple form 0 to pi.
        irregularity: maximum level of random point translation for irregular polygons.
        thickness: line width of the shapes.
        color: line color of the shapes.
        separated_channels: if True returns two images with one object per image.
    Returns:
        Numpy array."""
    
    # Array of size 128x128 filled with ones as values, to create an image with black color.
    full_img = np.zeros(shape=(64,64,3),dtype=np.int16)
    full_img[:] = (255, 255, 255)  # Changing the color of the image

    img1 = np.zeros(shape=(64,64,3),dtype=np.int16)
    img1[:] = (255, 255, 255)  # Changing the color of the image

    img2 = np.zeros(shape=(64,64,3),dtype=np.int16)
    img2[:] = (255, 255, 255)

    # Set up line color.
    color_a = (0,0,0) if color_a is None else color_a

    # By default make the color of the second object the same as the first one.
    if color_b is None:
        color_b = color_a

    # Get points.
    points_a, points_b, midpoint_2, radius_2 = svrt_1_points(
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

    # Reshaping according to opencv format
    poly_new_a = poly_a.reshape((-1,1,2))
    poly_new_b = poly_b.reshape((-1,1,2))

    # Draw.
    if not filled and not displace_vertices:
        # Full image 
        cv2.polylines(full_img,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
        cv2.polylines(full_img,[poly_new_b],isClosed=closed,color=color_a,thickness=thickness)
        # Individual
        cv2.polylines(img1,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
        cv2.polylines(img2,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)

    elif not filled and displace_vertices and category == 1:
        # Full image 
        cv2.polylines(full_img,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
        cv2.polylines(full_img,[poly_new_b],isClosed=closed,color=color_a,thickness=thickness)
        # Individual
        cv2.polylines(img1,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
        cv2.polylines(img2,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)

    elif not filled and displace_vertices and category == 0:
        # Full image 
        cv2.polylines(full_img,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
        cv2.polylines(full_img,[poly_new_b],isClosed=closed,color=color_a,thickness=thickness)
        off_x = random.sample(list(range(0, int(radius_2/2))), 4)
        off_y = random.sample(list(range(0, int(radius_2/2))), 4)
        full_img = scramble_poligon(full_img, midpoint=midpoint_2, radius=radius_2, off_x=off_x, off_y=off_y)
        # Individual
        cv2.polylines(img1,[poly_new_a],isClosed=closed,color=color_a,thickness=thickness)
        cv2.polylines(img2,[poly_new_b],isClosed=closed,color=color_b,thickness=thickness)
        img2 = scramble_poligon(img2, midpoint=midpoint_2, radius=radius_2, off_x=off_x, off_y=off_y)

    else:
        # Full image 
        cv2.fillPoly(full_img,[poly_new_a], color=color_a)
        cv2.fillPoly(full_img,[poly_new_b], color=color_a)
        # Individual
        cv2.fillPoly(img1, [poly_new_a], color=color_a)
        cv2.fillPoly(img2, [poly_new_b], color=color_b)
    
    # Return image(s).
    return full_img.astype('uint8'), img1.astype('uint8'), img2.astype('uint8')

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
    """Samples midpoints to arrows if sizes 5, 7 or 9 into a 64x64 image."""
    xs = random.sample(list(range(size*4, 64-size*4)), 2)
    ys = random.sample(list(range(size*4, 64-size*4)), 2)
    point_1 = [xs[0], ys[0]]
    point_2 = [xs[1], ys[1]]

    return point_1, point_2

def sample_midpoints_lines(sizes):
    """Samples midpoints of lines in a 64x64 image."""

    size_1, size_2 = sizes
    x_1 = random.sample(list(range(int(size_1/2)+2, 63-int(size_1/2+2))), 1)[0]
    y_1 = random.sample(list(range(int(size_1/2)+2, 63-int(size_1/2+2))), 1)[0]
    x_2 = random.sample(list(range(int(size_2/2)+2, 63-int(size_2/2+2))), 1)[0]
    y_2 = random.sample(list(range(int(size_2/2)+2, 63-int(size_2/2+2))), 1)[0]
    point_1 = (x_1, y_1)
    point_2 = (x_2, y_2)

    return point_1, point_2

def make_arrows_sd(
    category,
    continuous=False,
    line_width=1,
    hard_test=True
    ):
    """
    Args:
        category: 1 (same) or 0 (different).
        continuous: weather the line touches the available heads.
        line_width: line width in pixels.
    Returns:
        image (array).
    """
    
    # Background images.
    full_img = np.zeros(shape=(64,64,3),dtype=np.int16)
    full_img[:] = (255, 255, 255)  # Changing the color of the image

    img1 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img1[:] = (255, 255, 255)
    
    img2 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img2[:] = (255, 255, 255)

    # Sample sizes.
    size_1 = random.sample([4, 5, 6, 7], 1)[0]
    size_2 = random.sample([4, 5, 6, 7], 1)[0]
    
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
        size_2 = size_1
        rotation_2 = rotation_1
        head_a_2 = head_a_1
        head_b_2 = head_b_1
        shape_a_2 = shape_a_1
        shape_b_2 = shape_b_1
    
    if hard_test and category==0:
        size_2 = size_1
        rotation_2 = rotation_1
        # Ensure that the values of shape_a_1 and shape_b_1 are different.
        shape_a_1 = 'inverted' if shape_b_1 == 'inverted' else 'normal'
        # Ensure that the second arrow has the oposite head orientations.
        shape_a_2 = 'inverted' if shape_b_1 == 'normal' else 'normal'
        shape_b_2 = 'inverted' if  shape_a_1 == 'normal' else 'normal'

        if not head_a_1:
            head_b_2 = False
            head_a_2 = True
        if not head_b_1:
            head_a_2 = False
            head_b_2 = True
        if head_a_1 and head_b_1:
            head_a_2 = head_a_1
            head_b_2 = head_b_1

    # Get midpoints.
    midpoint_1, midpoint_2 = sample_midpoints_arrows(size=size_1)

    # Get arrow points.
    points_top_1, points_bottom_1, points_line_1 = get_arrow_points(radius=size_1,
                                                                    rotation=rotation_1,
                                                                    shape_a=shape_a_1, shape_b=shape_b_1,
                                                                    center=midpoint_1,
                                                                    continuous=continuous)

    points_top_2, points_bottom_2, points_line_2 = get_arrow_points(radius=size_2,
                                                                    rotation=rotation_2,
                                                                    shape_a=shape_a_2, shape_b=shape_b_2,
                                                                    center=midpoint_2,
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

    # Full image 
    if head_a_1:
        cv2.polylines(full_img,[poly_new_top_1],isClosed=True,color=(0, 0, 0),thickness=line_width)
    if head_b_1:
        cv2.polylines(full_img,[poly_new_bottom_1],isClosed=True,color=(0, 0, 0),thickness=line_width)
    cv2.line(full_img, points_line_1[0], points_line_1[1], (0, 0, 0), thickness=line_width)

    if head_a_2:
        cv2.polylines(full_img,[poly_new_top_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
    if head_b_2:
        cv2.polylines(full_img,[poly_new_bottom_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
    cv2.line(full_img, points_line_2[0], points_line_2[1], (0, 0, 0), thickness=line_width)
    
    # Individual
    if head_a_1:
        cv2.polylines(img1,[poly_new_top_1],isClosed=True,color=(0, 0, 0),thickness=line_width)
    if head_b_1:
        cv2.polylines(img1,[poly_new_bottom_1],isClosed=True,color=(0, 0, 0),thickness=line_width)
    cv2.line(img1, points_line_1[0], points_line_1[1], (0, 0, 0), thickness=line_width)

    if head_a_2:
        cv2.polylines(img2,[poly_new_top_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
    if head_b_2:
        cv2.polylines(img2,[poly_new_bottom_2],isClosed=True,color=(0, 0, 0),thickness=line_width)
    cv2.line(img2, points_line_2[0], points_line_2[1], (0, 0, 0), thickness=line_width)
    
    return full_img.astype('uint8'), img1.astype('uint8'), img2.astype('uint8')

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

def make_straingt_lines_sd(category, var_factor, line_thickness=1):
    # Background images
    full_img = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    full_img[:] = (255, 255, 255)

    img1 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img1[:] = (255, 255, 255)
    
    img2 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img2[:] = (255, 255, 255)

    # Sample sizes.
    sizes = random.sample(list(range(8, 36, 4)), 2)
    size_1 = sizes[0]
    size_2 = sizes[1]

    # Sample rotations.
    rotations = random.sample([0, 45, 90, 135], 2)
    rotation_1 = rotations[0]
    rotation_2 = rotations[1]

    # Assign size_2 and rotation_2 based on category and the variation factor.
    if category==1:
        size_2 = size_1
        rotation_2 = rotation_1
    
    if category==0:
        size_2 = size_1 if var_factor=='rotation' else size_2
        rotation_2 = rotation_1 if var_factor=='size' else rotation_2

    # Get midpoints.
    midpoint_1, midpoint_2 = sample_midpoints_lines(sizes=(size_1, size_2))

    # Get arrow points.
    points_line_1 = get_line_points(size=size_1, rotation=rotation_1, center=midpoint_1)
    points_line_2 = get_line_points(size=size_2, rotation=rotation_2, center=midpoint_2)

    # Draw!
    cv2.line(full_img, points_line_1[0], points_line_1[1], (0, 0, 0), thickness=line_thickness)
    cv2.line(full_img, points_line_2[0], points_line_2[1], (0, 0, 0), thickness=line_thickness)

    cv2.line(img1, points_line_1[0], points_line_1[1], (0, 0, 0), thickness=line_thickness)
    cv2.line(img2, points_line_2[0], points_line_2[1], (0, 0, 0), thickness=line_thickness)
    
    return full_img.astype('uint8'), img1.astype('uint8'), img2.astype('uint8')

# Rectanngles
def make_rectangles_sd(category, line_thickness=1):
    # Background images
    full_img = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    full_img[:] = (255, 255, 255)

    img1 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img1[:] = (255, 255, 255)
    
    img2 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img2[:] = (255, 255, 255)

    # Sample constant dimension.
    const_dim = 'x' if random.random() > 0.5 else 'y'

    # Sample sizes.
    if const_dim == 'y':
        sizes_x = random.sample(list(range(8, 36, 4)), 2)
        size_x_1 = sizes_x[0]
        size_x_2 = sizes_x[1]

        size_y_1 = random.sample(list(range(8, 36, 4)), 1)[0]
        size_y_2 = size_y_1

        # Assign size_x_2 based on category.
        if category==1:
            size_x_2 = size_x_1
    
    elif const_dim == 'x':
        sizes_y = random.sample(list(range(8, 36, 4)), 2)
        size_y_1 = sizes_y[0]
        size_y_2 = sizes_y[1]

        size_x_1 = random.sample(list(range(8, 36, 4)), 1)[0]
        size_x_2 = size_x_1

        # Assign size_y_2 based on category.
        if category==1:
            size_y_2 = size_y_1

    # Sample start and end points.
    x_1 = random.sample(list(range(2, 63-(size_x_1+2))), 1)[0]
    y_1 = random.sample(list(range(2, 63-(size_y_1+2))), 1)[0]
    x_2 = random.sample(list(range(2, 63-(size_x_2+2))), 1)[0]
    y_2 = random.sample(list(range(2, 63-(size_y_2+2))), 1)[0]
    start_point_1 = (x_1, y_1)
    start_point_2 = (x_2, y_2)
    end_point_1 = (x_1+size_x_1, y_1+size_y_1)
    end_point_2 = (x_2+size_x_2, y_2+size_y_2)

    # Draw squares
    full_img = cv2.rectangle(full_img, start_point_1, end_point_1, (0, 0, 0), line_thickness)
    full_img = cv2.rectangle(full_img, start_point_2, end_point_2, (0, 0, 0), line_thickness)

    img1 = cv2.rectangle(img1, start_point_1, end_point_1, (0, 0, 0), line_thickness)
    img2 = cv2.rectangle(img2, start_point_2, end_point_2, (0, 0, 0), line_thickness)
    
    return full_img.astype('uint8'), img1.astype('uint8'), img2.astype('uint8')

# Connected squares 
def make_connected_open_squares(category, line_width=1, is_closed=False):
    # Background images.
    full_img = np.zeros(shape=(64,64,3),dtype=np.int16)
    full_img[:] = (255, 255, 255)  # Changing the color of the image

    img1 = np.zeros(shape=(64, 64, 3), dtype=np.int16)
    img1[:] = (255, 255, 255)
    
    img2 = np.zeros(shape=(64, 64, 3), dtype=np.int16)
    img2[:] = (255, 255, 255)

    # Sample segment size.
    size = random.randint(5, 12)

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

    points_b = [
        [0, size],
        [0, 2*size],
        [size, 2*size],
        [size, size],
        [size, 0],
        [2*size, 0],
        [2*size, size]
        ]
    # Assign points based on category.
    if category==1:
            points_b = points_a

    # Sample translations and apply.
    translation_a = [np.random.randint(1, 63-size*2), np.random.randint(1, 63-size*2)]
    translation_b = [np.random.randint(1, 63-size*2), np.random.randint(1, 63-size*2)]
    points_a = [[sum(pair) for pair in zip(point, translation_a)] for point in points_a]
    points_b = [[sum(pair) for pair in zip(point, translation_b)] for point in points_b]

    # Assigning sides to polygon
    poly_a = np.array(points_a,dtype=np.int32)
    poly_b = np.array(points_b,dtype=np.int32)

    # Reshaping according to opencv format
    poly_new_a = poly_a.reshape((-1,1,2))
    poly_new_b = poly_b.reshape((-1,1,2))

    # Full image
    cv2.polylines(full_img,[poly_new_a],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
    cv2.polylines(full_img,[poly_new_b],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
    # individual images
    cv2.polylines(img1,[poly_new_a],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)
    cv2.polylines(img2,[poly_new_b],isClosed=is_closed,color=(0, 0, 0),thickness=line_width)

    return full_img.astype('uint8'), img1.astype('uint8'), img2.astype('uint8')

# Connected circles
def make_connected_circles(category, line_thickness=1):
    # Background images
    full_img = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    full_img[:] = (255, 255, 255)

    img1 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img1[:] = (255, 255, 255)
    
    img2 = np.zeros(shape=(64, 64, 3), dtype=np.uint8)
    img2[:] = (255, 255, 255)
    
    # Get small and big radious.
    radii = random.sample(list(range(4, 16, 4)), 2)
    radii.sort()
    radius_small = radii[0]
    radius_big = radii[1]
    
    # Sample ordering of circes: small-big vs big-small.
    order_1 = 'sb' if random.random() < 0.5 else 'bs'
    order_2 = 'bs' if order_1 == 'sb' else 'sb'
    
    # Assign radii based on order.
    radius_1_a = radius_small if order_1 == 'sb' else radius_big
    radius_1_b = radius_big if order_1 == 'sb' else radius_small
    radius_2_a = radius_small if order_2 == 'sb' else radius_big
    radius_2_b = radius_big if order_2 == 'sb' else radius_small

    # Assign radius_big based on category.
    if category==1:
            radius_2_a = radius_1_a
            radius_2_b = radius_1_b
    
    # Sample midpoints.
    mpdt_1 = (
        np.random.randint(radius_big+2, 62-radius_big),
        np.random.randint(radius_1_a+radius_1_b+2, 62-(radius_1_a+radius_1_b))
        )
    
    mpdt_2 = (
        np.random.randint(radius_big+2, 62-radius_big),
        np.random.randint(radius_2_a+radius_2_b+2, 62-(radius_2_a+radius_2_b))
        )

    mdpt_1_a = (mpdt_1[0], mpdt_1[1]-radius_1_b)
    mdpt_1_b = (mpdt_1[0], mpdt_1[1]+radius_1_a)

    mdpt_2_a = (mpdt_2[0], mpdt_2[1]-radius_2_b)
    mdpt_2_b = (mpdt_2[0], mpdt_2[1]+radius_2_a)


    # Draw circles.
    full_img = cv2.circle(full_img, mdpt_1_a, radius_1_a, (0, 0, 0), line_thickness)
    full_img = cv2.circle(full_img, mdpt_1_b, radius_1_b, (0, 0, 0), line_thickness)
    full_img = cv2.circle(full_img, mdpt_2_a, radius_2_a, (0, 0, 0), line_thickness)
    full_img = cv2.circle(full_img, mdpt_2_b, radius_2_b, (0, 0, 0), line_thickness)

    img1 = cv2.circle(img1, mdpt_1_a, radius_1_a, (0, 0, 0), line_thickness)
    img1 = cv2.circle(img1, mdpt_1_b, radius_1_b, (0, 0, 0), line_thickness)

    img2 = cv2.circle(img2, mdpt_2_a, radius_2_a, (0, 0, 0), line_thickness)
    img2 = cv2.circle(img2, mdpt_2_b, radius_2_b, (0, 0, 0), line_thickness)
    
    return full_img.astype('uint8'), img1.astype('uint8'), img2.astype('uint8')

# Helper function to catch bad sampless
def compare_xy(point_1, point_2):
    # Is the lower object to the right of the upper object?
    lower_obj = point_1 if point_1[1] >= point_2[1] else point_2
    upper_obj = point_1 if lower_obj is point_2 else point_2
    comparison = 1 if lower_obj[0] >= upper_obj[0] else 0
    return comparison

def get_shapes_info_old(img, scrambled_negative=False, draw_b_rects=False):
    """Returns info for multi-task learning. Obj1 and obj2 are determined
    by a biased distance from the origin (d = 1.1*x^2 + y^2).
    Args:
        img: image (np.array)
        scrambled_negative: whether img is a negative example from the scrambled condition.
    Returns:
        coor1: x, y coordinate of the center of obj1.
        coor2: x, y coordinate of the center of obj2.
        relative_position: whether the lower object is to the right of the upper object.
        flag: True if the objects are touching. Used to discard the sample."""

    ## Get coordinates.
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours, obtain bounding box.
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # this is to deal with different opencv versions.

    # Get bounding boxes.
    b_rects = []
    for c in cnts:
        b_rects.append(cv2.boundingRect(c))
    if scrambled_negative:
        # Get the bigest bounding box.
        new_b_rects = []
        bigest = max(enumerate(b_rects),key=lambda x: x[1][2]*x[1][3])[1]
        new_b_rects.append(bigest)
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
        # If there are none smaller rects, set b_rects to a list of 3 bounding rects,
        # such that it activates the bad sample flag
        else:
            b_rects = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    
    # Define obj1 and obj2.
    b_rect_1 = min(enumerate(b_rects),key=lambda x: 1.1*x[1][0]**2 + x[1][1]**2)[1]
    b_rect_2 = max(enumerate(b_rects),key=lambda x: 1.1*x[1][0]**2 + x[1][1]**2)[1]
    mp_1 = np.array([b_rect_1[0]+b_rect_1[2]/2, b_rect_1[1]+b_rect_1[3]/2])
    mp_2 = np.array([b_rect_2[0]+b_rect_2[2]/2, b_rect_2[1]+b_rect_2[3]/2])
    
    # Get relations.
    relative_position = compare_xy(mp_1, mp_2)

    # Get flag.
    flag = False if len(b_rects) == 2 else True

    if draw_b_rects:
        # Draw bounding rects.
        for b in b_rects:
            x,y,w,h = b
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
        # Draw midpoints.
        cv2.line(img, (int(mp_1[0]), int(mp_1[1])), (int(mp_1[0]), int(mp_1[1])), (255,0,0), 2)
        cv2.line(img, (int(mp_2[0]), int(mp_2[1])), (int(mp_2[0]), int(mp_2[1])), (255,0,0), 2)
        # show image.
        # plt.imshow(img)
        # plt.show()
    
    return np.concatenate((mp_1, mp_2), axis=0), relative_position, flag

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
        # Check if at least one cnt is closed, if not return inmediatly with flag
        closed_test_list = [cv2.contourArea(x) > cv2.arcLength(x, True) for x in cnts]
        if not any(closed_test_list):
            flag = True
            return np.array([0, 0, 0, 0]), 0, flag

        # Get the bigest bounding box.
        new_b_rects = []
        bigest = max(enumerate(b_rects),key=lambda x: x[1][2]*x[1][3])[1]
        new_b_rects.append(bigest)
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
    flag = False if len(b_rects) == 2 else True

    # Chech if shapes are touching the image border
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
    
    # Define obj1 and obj2.
    b_rect_1 = min(enumerate(b_rects),key=lambda x: 1.1*x[1][0]**2 + x[1][1]**2)[1]
    b_rect_2 = max(enumerate(b_rects),key=lambda x: 1.1*x[1][0]**2 + x[1][1]**2)[1]
    mp_1 = np.array([b_rect_1[0]+b_rect_1[2]/2, b_rect_1[1]+b_rect_1[3]/2])
    mp_2 = np.array([b_rect_2[0]+b_rect_2[2]/2, b_rect_2[1]+b_rect_2[3]/2])
    
    # Get relations.
    relative_position = compare_xy(mp_1, mp_2)
    
    if draw_b_rects:
        # Draw bounding rects.
        for b in b_rects:
            x,y,w,h = b
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
        # Show image
        # plt.imshow(img)
        # plt.show()
    
    return np.concatenate((mp_1, mp_2), axis=0), relative_position, flag

def build_complementary_samples(
    base_category,
    base_full_array,
    base1_array,
    base2_array,
    same_full_array, 
    same1_array,
    same2_array,
    diff_full_array,
    diff1_array,
    diff2_array
    ):
    # Coordinates
    left_1, top_1 = 32, 0
    left_2, top_2 = 0, 64
    left_3, top_3 = 64, 64

    # Get individual images
    base_full = Image.fromarray(base_full_array)
    base1 = Image.fromarray(base1_array)
    base2 = Image.fromarray(base2_array)
    if base_category == 0: # 'diff' base
        match_full = Image.fromarray(diff_full_array)
        match1 = Image.fromarray(diff1_array)
        match2 = Image.fromarray(diff2_array)
        nonmatch_full = Image.fromarray(same_full_array)
        nonmatch1 = Image.fromarray(same1_array)
        nonmatch2 = Image.fromarray(same2_array)
    elif base_category == 1: # 'same' base
        match_full = Image.fromarray(same_full_array)
        match1 = Image.fromarray(same1_array)
        match2 = Image.fromarray(same2_array)
        nonmatch_full = Image.fromarray(diff_full_array)
        nonmatch1 = Image.fromarray(diff1_array)
        nonmatch2 = Image.fromarray(diff2_array)
    else:
        raise ValueError('base_category has to be 0 or 1!')

    # "match/non-match" version
    mn_full = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_full.paste(base_full, (left_1, top_1))
    mn_full.paste(match_full, (left_2, top_2))
    mn_full.paste(nonmatch_full, (left_3, top_3))

    mn_base1 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_base1.paste(base1, (left_1, top_1))
    mn_base2 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_base2.paste(base2, (left_1, top_1))

    mn_left1 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_left1.paste(match1, (left_2, top_2))
    mn_left2 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_left2.paste(match2, (left_2, top_2))

    mn_right1 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_right1.paste(nonmatch1, (left_3, top_3))
    mn_right2 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    mn_right2.paste(nonmatch2, (left_3, top_3))
    
    # match label: 0 (left)
    mn_sample = (mn_full, [mn_base1, mn_base2, mn_left1, mn_left2, mn_right1, mn_right2], 0)

    # "non-match/match" version
    nm_full = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_full.paste(base_full, (left_1, top_1))
    nm_full.paste(nonmatch_full, (left_2, top_2))
    nm_full.paste(match_full, (left_3, top_3))
    
    nm_base1 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_base1.paste(base1, (left_1, top_1))
    nm_base2 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_base2.paste(base2, (left_1, top_1))

    nm_left1 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_left1.paste(nonmatch1, (left_2, top_2))
    nm_left2 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_left2.paste(nonmatch2, (left_2, top_2))

    nm_right1 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_right1.paste(match1, (left_3, top_3))
    nm_right2 = Image.new(base_full.mode, (128, 128), (255, 255, 255))
    nm_right2.paste(match2, (left_3, top_3))

    # match label: 1 (right)
    nm_sample = (nm_full, [nm_base1, nm_base2, nm_left1, nm_left2, nm_right1, nm_right2], 1)

    return mn_sample, nm_sample

# Functions to generate deterministic samples
def make_rms_irregular_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'irregular' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = svrt_1_img(base_category, regular=False, color_a=(0,0,0), sides=None, thickness=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = svrt_1_img(1, regular=False, color_a=(0,0,0), sides=None, thickness=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = svrt_1_img(0, regular=False, color_a=(0,0,0), sides=None, thickness=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_regular_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'regular' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = svrt_1_img(base_category, regular=True, color_a=(0,0,0), sides=None, thickness=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        # if not bad_sample:
        #     pil_image = Image.fromarray(base_array)
        #     pil_image.show()
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = svrt_1_img(1, regular=True, color_a=(0,0,0), sides=None, thickness=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = svrt_1_img(0, regular=True, color_a=(0,0,0), sides=None, thickness=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_open_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'open' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        r1 = np.random.randint(10, 14) # np.random.randint(10, 40)
        r2 = np.random.randint(10, 14) #radius_1 #if category==1 else np.random.randint(10, 40)
        base_full, base1, base2 = svrt_1_img(base_category, regular=False, color_a=(0,0,0), sides=None, thickness=1, closed=False, radii=(r1, r2))
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        r1 = np.random.randint(10, 14) # np.random.randint(10, 40)
        r2 = np.random.randint(10, 14) #radius_1 #if category==1 else np.random.randint(10, 40)
        same_full, same1, same2 = svrt_1_img(1, regular=False, color_a=(0,0,0), sides=None, thickness=1, closed=False, radii=(r1, r2)) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        r1 = np.random.randint(10, 14) # np.random.randint(10, 40)
        r2 = np.random.randint(10, 14) #radius_1 #if category==1 else np.random.randint(10, 40)
        diff_full, diff1, diff2 = svrt_1_img(0, regular=False, color_a=(0,0,0), sides=None, thickness=1, closed=False, radii=(r1, r2)) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_wider_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'wider' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = svrt_1_img(base_category, regular=False, color_a=(0,0,0), sides=None, thickness=2)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = svrt_1_img(1, regular=False, color_a=(0,0,0), sides=None, thickness=2) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = svrt_1_img(0, regular=False, color_a=(0,0,0), sides=None, thickness=2) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_scrambled_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'scrambled' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        scrambled_negative = True if base_category == 0 else False
        base_full, base1, base2 = svrt_1_img(
            category=base_category, 
            regular=True, 
            color_a=(0,0,0), 
            sides=None, 
            thickness=1, 
            displace_vertices=True
            )
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=scrambled_negative, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = svrt_1_img(
            category=1, 
            regular=True, 
            color_a=(0,0,0), 
            sides=None, 
            thickness=1, 
            displace_vertices=True
            ) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = svrt_1_img(
            category=0, 
            regular=True, 
            color_a=(0,0,0), 
            sides=None, 
            thickness=1, 
            displace_vertices=True
            ) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=True, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_randomcolor_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'random color' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        color = tuple(np.random.randint(251, size=3))
        color = (int(color[0]), int(color[1]), int(color[2]))
        base_full, base1, base2 = svrt_1_img(base_category, regular=False, color_a=color, sides=None, thickness=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        color = tuple(np.random.randint(251, size=3))
        color = (int(color[0]), int(color[1]), int(color[2]))
        same_full, same1, same2 = svrt_1_img(1, regular=False, color_a=color, sides=None, thickness=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        color = tuple(np.random.randint(251, size=3))
        color = (int(color[0]), int(color[1]), int(color[2]))
        diff_full, diff1, diff2 = svrt_1_img(0, regular=False, color_a=color, sides=None, thickness=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_filled_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'filled' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = svrt_1_img(base_category, regular=False, color_a=(0,0,0), sides=None, thickness=1, filled=True)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = svrt_1_img(1, regular=False, color_a=(0,0,0), sides=None, thickness=1, filled=True) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = svrt_1_img(0, regular=False, color_a=(0,0,0), sides=None, thickness=1, filled=True) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_opensquares_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'open squares' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = make_connected_open_squares(base_category, line_width=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = make_connected_open_squares(1, line_width=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = make_connected_open_squares(0, line_width=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_arrows_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'arrows' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = make_arrows_sd(base_category, continuous=True, line_width=1, hard_test=True)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = make_arrows_sd(1, continuous=True, line_width=1, hard_test=True) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = make_arrows_sd(0, continuous=True, line_width=1, hard_test=True) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_rectangles_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'rectangles' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = make_rectangles_sd(base_category, line_thickness=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = make_rectangles_sd(1, line_thickness=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = make_rectangles_sd(0, line_thickness=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_straightlines_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'straight lines' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = make_straingt_lines_sd(base_category, var_factor='size', line_thickness=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = make_straingt_lines_sd(1, var_factor='size', line_thickness=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = make_straingt_lines_sd(0, var_factor='size', line_thickness=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_closedsquares_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'connected squares' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = make_connected_open_squares(base_category, line_width=1, is_closed=True)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = make_connected_open_squares(1, line_width=1, is_closed=True) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = make_connected_open_squares(0, line_width=1, is_closed=True) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def make_rms_connectedcircles_sample(base_category):
    """Makes two complementary samples of the RMTS task in the 'connected circles' condition.
    Args:
        base_category: 0 (diff) or 1 (same).
    """
    while True:
        base_full, base1, base2 = make_connected_circles(base_category, line_thickness=1)
        _, _, bad_sample = get_shapes_info(base_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        same_full, same1, same2 = make_connected_circles(1, line_thickness=1) # same
        _, _, bad_sample = get_shapes_info(same_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    while True:
        diff_full, diff1, diff2 = make_connected_circles(0, line_thickness=1) # different
        _, _, bad_sample = get_shapes_info(diff_full, scrambled_negative=False, draw_b_rects=False)
        if bad_sample:
            pass
        else:
            break
    # Build samples
    return build_complementary_samples(
        base_category=base_category,
        base_full_array=base_full,
        base1_array=base1,
        base2_array=base2,
        same_full_array=same_full, 
        same1_array=same1,
        same2_array=same2,
        diff_full_array=diff_full,
        diff1_array=diff1,
        diff2_array=diff2
        )

def check_path(path):
    """Creates a directory if it doesn't exist yet"""
    if not os.path.exists(path):
        os.mkdir(path)

# Main data generation function
def make_rmts_datasets(train_size, val_size, test_size):
    # Gather all ds generators
    datasets = [
        make_rms_irregular_sample,
        # make_rms_regular_sample,
        # make_rms_open_sample,
        # make_rms_wider_sample,
        make_rms_scrambled_sample,
        # make_rms_randomcolor_sample,
        # make_rms_filled_sample,
        # make_rms_opensquares_sample,
        # make_rms_arrows_sample,
        # make_rms_rectangles_sample,
        # make_rms_straightlines_sample,
        # make_rms_closedsquares_sample,
        # make_rms_connectedcircles_sample
        ]
    # Get all ds names
    ds_names = [
        'irregular_rmts',
        # 'regular_rmts',
        # 'open_rmts', 
        # 'wider_rmts', 
        'scrambled_srmts',
        # 'random_rmts',
        # 'filled_rmts',
        # 'lines_rmts',
        # 'arrows_rmts',
        # 'rectangles_rmts',
        # 'slines_rmts',
        # 'csquares_rmts',
        # 'ccircles_rmts'
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
            ds_file_header = ['ID', 'base', 'label']
            # Generate data
            counter = 0
            # Open the file in write mode
            with open(ds_file, 'w') as f:
                # Create the csv writer
                writer = csv.writer(f)
                # Write header to the csv file
                writer.writerow(ds_file_header)
                for i in range(size):
                    # Generate data: 
                    # base_category: 0 (diff) or 1 (same).
                    # match_value: 0 (left) or 1 (right).
                    base_category = 0 if random.uniform(0, 1) < 0.5 else 1 
                    mn_sample, nm_sample = gen(base_category)
                    # Save each sample
                    full_img, image_list, match_value = mn_sample
                    row = [counter, base_category, match_value] 
                    full_img.save(f'{ds_dir}/{counter}.png')
                    writer.writerow(row)
                    counter += 1

                    full_img, image_list, match_value = nm_sample
                    row = [counter, base_category, match_value] 
                    full_img.save(f'{ds_dir}/{counter}.png')
                    writer.writerow(row)
                    counter += 1
                    
    return

if __name__ == '__main__':
    # Parameters
    # Total: 280_000. Train (70%): 196_000, val (10%): 28_000, test (20%): 56_000
    TRAIN_SIZE = 98000 # double
    VAL_SIZE = 14000 # double
    TEST_SIZE = 28000 # double
    
    # Generate datasets
    make_rmts_datasets(
        train_size=TRAIN_SIZE, 
        val_size=VAL_SIZE, 
        test_size=TEST_SIZE
        )
    print('Done!')
