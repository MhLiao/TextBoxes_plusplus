import sys
import math
import os
from PIL import Image,ImageDraw
import numpy as np
import cv2
import shapely
from shapely.geometry import Polygon

def general_crop(image, tile):
    """Crop the image giving a tile.
    Args:
        image: Image to be crop, [h, w, c].
        tile: [p_0, p_1, p_2, p_3] (clockwise).

    Returns:
        cropped: Patch corresponding to the tile.

    Raises:
        ZeroDivisionError: x[1] == x[0] or x[2] == x[3].
    """
    x = [p[0] for p in tile]
    y = [p[1] for p in tile]
    # phase1:shift the center of patch to image center
    x_center = int(round(sum(x) / 4))
    y_center = int(round(sum(y) / 4))
    im_center = [int(round(coord / 2)) for coord in image.shape[:2]]
    shift = [im_center[0] - y_center, im_center[1] - x_center]
    M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    height, width = image.shape[:2]
    im_shift = cv2.warpAffine(image, M, (width, height))

    # phase2:imrote the im_shift to regular the box
    bb_width = (math.sqrt((y[1] - y[0]) ** 2 + (x[1] - x[0]) ** 2) +
                math.sqrt((y[3] - y[2]) ** 2 + (x[3] - x[2]) ** 2)) / 2
    bb_height = (math.sqrt((y[3] - y[0]) ** 2 + (x[3] - x[0]) ** 2) +
                 math.sqrt((y[2] - y[1]) ** 2 + (x[2] - x[1]) ** 2)) / 2
    horiz = True
    if bb_width > bb_height:  # main direction is horizental
        tan = ((y[1] - y[0]) / float(x[1] - x[0] + 1e-8) +
               (y[2] - y[3]) / float(x[2] - x[3] + 1e-8)) / 2
        degree = math.atan(tan) / math.pi * 180
    else:  # main direction is vertical
        tan = ((y[1] - y[2]) / float(x[1] - x[2] + 1e-8) +
               (y[0] - y[3]) / float(x[0] - x[3] + 1e-8)) / 2
        degree = math.atan(tan) / math.pi * 180 - np.sign(tan) * 90
        horiz = False
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), degree, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (nW / 2) - width//2
    rotation_matrix[1, 2] += (nH / 2) - height//2

    im_rotate = cv2.warpAffine(im_shift, rotation_matrix, (nW, nH))
    # phase3:crop the box out.
    (newCX, newCY) = (nW // 2, nH // 2)
    x_min = max(newCX - int(round(bb_width / 2)), 0)
    x_max = min(newCX + int(round(bb_width / 2)), nW)
    y_min = max(newCY - int(round(bb_height / 2)), 0)
    y_max = min(newCY + int(round(bb_height / 2)), nH)
    print([x_min,y_min,x_max,y_max])
    print([nW,nH])
    return im_rotate[y_min:y_max, x_min:x_max, :], horiz

def general_crop_expand(image, tile):
    """Crop the image giving a tile.
    Args:
        image: Image to be crop, [h, w, c].
        tile: [p_0, p_1, p_2, p_3] (clockwise).

    Returns:
        cropped: Patch corresponding to the tile.

    Raises:
        ZeroDivisionError: x[1] == x[0] or x[2] == x[3].
    """
    x = [p[0] for p in tile]
    y = [p[1] for p in tile]
    # phase1:shift the center of patch to image center
    x_center = int(round(sum(x) / 4))
    y_center = int(round(sum(y) / 4))
    im_center = [int(round(coord / 2)) for coord in image.shape[:2]]
    shift = [im_center[0] - y_center, im_center[1] - x_center]
    M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    height, width = image.shape[:2]
    im_shift = cv2.warpAffine(image, M, (width, height))

    # phase2:imrote the im_shift to regular the box
    bb_width = (math.sqrt((y[1] - y[0]) ** 2 + (x[1] - x[0]) ** 2) +
                math.sqrt((y[3] - y[2]) ** 2 + (x[3] - x[2]) ** 2)) / 2
    bb_height = (math.sqrt((y[3] - y[0]) ** 2 + (x[3] - x[0]) ** 2) +
                 math.sqrt((y[2] - y[1]) ** 2 + (x[2] - x[1]) ** 2)) / 2
    horiz = True
    if bb_width > bb_height:  # main direction is horizental
        tan = ((y[1] - y[0]) / float(x[1] - x[0] + 1e-8) +
               (y[2] - y[3]) / float(x[2] - x[3] + 1e-8)) / 2
        degree = math.atan(tan) / math.pi * 180
    else:  # main direction is vertical
        tan = ((y[1] - y[2]) / float(x[1] - x[2] + 1e-8) +
               (y[0] - y[3]) / float(x[0] - x[3] + 1e-8)) / 2
        degree = math.atan(tan) / math.pi * 180 - np.sign(tan) * 90
        horiz = False
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), degree, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (nW / 2) - width//2
    rotation_matrix[1, 2] += (nH / 2) - height//2

    im_rotate = cv2.warpAffine(im_shift, rotation_matrix, (nW, nH))
    # phase3:crop the box out.
    (newCX, newCY) = (nW // 2, nH // 2)
    expand_dist = bb_height / 6.
    x_min = max(newCX - int(round(bb_width / 2) - expand_dist), 0)
    x_max = min(newCX + int(round(bb_width / 2) + expand_dist), nW)
    y_min = max(newCY - int(round(bb_height / 2) - expand_dist), 0)
    y_max = min(newCY + int(round(bb_height / 2) + expand_dist), nH)
    print([x_min,y_min,x_max,y_max])
    print([nW,nH])
    return im_rotate[y_min:y_max, x_min:x_max, :], horiz

def crop_image(image_path, detection_results, crop_dir):
    if not os.path.isdir(crop_dir):
        os.mkdir(crop_dir)
    img=Image.open(image_path)
    img_name=image_path.split('/')[-1]
    width, height=img.size
    sub_dir=crop_dir+img_name.split('.')[0]+'/'
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    index=0
    for result in detection_results:
        index = index + 1
        crop_name=str(index)+'.jpg'
        save_crop_path=os.path.join(sub_dir,crop_name)
        if os.path.isfile(save_crop_path):
            continue
        score = result[-1]
        x1 = result[0]
        y1 = result[1]
        x2 = result[2]
        y2 = result[3]
        x3 = result[4]
        y3 = result[5]
        x4 = result[6]
        y4 = result[7]
        tile=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        if (x1,y1)==(x4,y4):
            continue
        if(Polygon(tile).area==0):
            continue
        crop,hori=general_crop_expand(np.array(img),tile)
        crop=Image.fromarray(crop)
        crop.save(save_crop_path)
        

