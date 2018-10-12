import os, random, sys
import numpy as np
import cv2
from scipy import ndimage

IMAGE_DIRS = ['comics', 'GMG']
IMAGE_W = 768
IMAGE_H = 256
USE_EDGES = False
IS_MINI = False
if IS_MINI:
	IMAGE_W /= 4
	IMAGE_H /= 4

all_comics = []
num_comics = 0

print "Loading Images..."
for dir in IMAGE_DIRS:
	for file in os.listdir(dir):
		path = dir + "/" + file
		
		#Only attempt to load standard image formats
		path_split = path.split('.')
		if len(path_split) < 2: continue
		if path_split[-1] not in ['bmp', 'gif', 'png', 'jpg', 'jpeg']:
			continue
		
		#Make sure image is valid and not corrupt
		img = ndimage.imread(path)
		if img is None:
			assert(False)
		if len(img.shape) != 3 or img.shape[2] != 3:
			continue
		
		#Make sure this is a standard 3-panel comic, not a sunday one
		ratio = float(img.shape[1]) / float(img.shape[0])
		if ratio < 3.17 or ratio > 3.60:
			continue

		#Scale all comics to a uniform size
		img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)
		
		if USE_EDGES:
			img = np.where(np.amax(img, axis=2) < 128, 255, 0).astype(np.uint8)

		panel_1 = img[:,0*IMAGE_W/3:1*IMAGE_W/3]
		panel_2 = img[:,1*IMAGE_W/3:2*IMAGE_W/3]
		panel_3 = img[:,2*IMAGE_W/3:3*IMAGE_W/3]

		if USE_EDGES:
			cur_comic = np.empty((3, 1, IMAGE_H, IMAGE_W/3), dtype=np.uint8)
			cur_comic[0][0] = panel_1
			cur_comic[1][0] = panel_2
			cur_comic[2][0] = panel_3
		else:
			cur_comic = np.empty((3, 3, IMAGE_H, IMAGE_W/3), dtype=np.uint8)
			cur_comic[0] = np.transpose(panel_1, (2, 0, 1))
			cur_comic[1] = np.transpose(panel_2, (2, 0, 1))
			cur_comic[2] = np.transpose(panel_3, (2, 0, 1))

		all_comics.append(cur_comic)
		
		num_comics += 1
		if num_comics % 10 == 0:
			sys.stdout.write('\r')
			sys.stdout.write(str(num_comics))
			sys.stdout.flush()
print "\nLoaded " + str(num_comics) + " comics."

print "Saving..."
all_comics = np.stack(all_comics, axis=0)
fname = 'data/comics'
if USE_EDGES:
	fname += '_edges'
if IS_MINI:
	fname += '_mini'
np.save(fname + '.npy', all_comics)

print "Done"