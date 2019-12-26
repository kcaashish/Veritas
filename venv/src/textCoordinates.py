import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os

img = cv2.imread("images/images1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#kernel = np.ones((3, 3), np.uint8)
#imgMorph = cv2.erode(imgContrast, kernel, iterations=1)


kernel = np.ones((2, 2), np.uint8)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
blur = cv2.medianBlur(thresh, 5)
dilated = cv2.dilate(blur, kernel, iterations=1)

#gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)


ctrs, hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_contours):
	x, y, w, h = cv2.boundingRect(ctr)

	roi = img[y:y+h, x:x+w]

	if w >15 and h > 15:
		cv2.imwrite("extracted/{}.jpg".format(i), roi)
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("original Image", img)
#cv2.imshow("Morphed Image", imgMorph)
#cv2.imshow("Thresholded", thresh)
cv2.imshow("Dilated", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
