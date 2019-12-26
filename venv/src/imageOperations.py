import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def main():
	IMAGEPATH = "images/"
	THRESHPATH = "thresh/"
	if not os.path.exists(THRESHPATH):
		os.makedirs(THRESHPATH)
	files = os.listdir(IMAGEPATH)

	for file in files:
		print(file)
		img = cv2.imread(IMAGEPATH+file, 0)
		ret, threshImg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

		comp = cv2.connectedComponentsWithStats(threshImg)

		cv2.imwrite(THRESHPATH + file, threshImg)

if __name__ == '__main__':
	main()
