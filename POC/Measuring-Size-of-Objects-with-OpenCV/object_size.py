# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

# py object_size.py --image images/Nut_7a.png --width 3.8

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# Colors are in BGR format
COLOR_BLUE		= (255, 0  , 0  )
COLOR_GREEN 	= (0  , 255, 0  )
COLOR_RED		= (0  , 0  , 255)
COLOR_MAGENTA 	= (255, 0  , 255)
COLOR_WHITE		= (255, 255, 255)

# Detection parameters
CIRCLE_RADIUS_DELTA_CM = 1 #delta between Width and Length to be considered a circle
CONTAINER_RADIUS_ARR_CM = np.array([12.59, 14.46, 16.41, 18.24, 22.4]) #set the different raius of the object to find here, the order is important ! (the index is used to tell which object is found)
CONTAINER_RADIUS_DELTA_CM = 1 #delta beteween the calculated radius and the array to be considered valid
CONTAINER_COLOR_ARR_BGR = np.array([
    [168, 182, 170],    # Container 0 light green (12.59 cm)
    [158, 172, 160],    # Container 1 light green (14.46 cm)
    [158, 172, 150],    # Container 2 light green (16.41 cm)
    [146, 161, 142],  	# Container 3 light green (18.24 cm)
    [36, 33, 32]   		# Container 4 dark gray   (22.4 cm)
])
CONTAINER_COLOR_TOLERANCE = 30

# Display object name and radius
REF_DISP_PX_OFFSET = 25 # Reference to the Y axis of the top left point
OBJECT_DISP_PX_OFFSET_FIRST = -10 # Reference to the Y axis of the bottom left point
OBJECT_DISP_PX_OFFSET_SECOND = OBJECT_DISP_PX_OFFSET_FIRST - 25 # 25 is the size of the text above and a little space
OBJECT_DISP_PX_OFFSET_THIRD = OBJECT_DISP_PX_OFFSET_SECOND - 25

# Only display the final result or all the steps
DISPLAY_STEP_BY_STEP = True
# Use gray levels instead of RGB levels
USE_GRAY_LEVELS = False

# Global variables for threshold callback
edged = 0
threshold = 50
g_contours = 0
g_hierarchy = 0

# Calculate midpoint
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Show image and wait for key to continue
def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()

# Callbacck to define the threshold to use and display it
def thresh_callback(val):
	global threshold 
	threshold = val
	global edged
	global g_contours
	global g_hierarchy
    # Detect edges using Canny
	edged = cv2.Canny(blur, threshold, threshold * 2)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
    # Find contours
	g_contours, g_hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
	drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)
	cv2.drawContours(drawing, g_contours, -1, COLOR_RED, 2, cv2.LINE_8, g_hierarchy, 0)
	# Show in a window
	cv2.imshow('Contours', drawing)
	cv2.imshow("Dilate and erode image", edged)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Enchance contrast HSV : Hue, Saturation, Value (Brightness)
if USE_GRAY_LEVELS:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if DISPLAY_STEP_BY_STEP :
		show_image("Gray Image", gray, False)
	blur = cv2.GaussianBlur(gray, (7, 7), 0)
else:
	if DISPLAY_STEP_BY_STEP :
		show_image("RGB Image", image, False)
	blur = cv2.GaussianBlur(image, (7, 7), 0)
if DISPLAY_STEP_BY_STEP :
	show_image("Blur Image", blur, True)

#### adpativeThreshold detection
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
# show_image("Adaptive Mean Threshold Image", thresh, False)
# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
# show_image("Adaptive Gaussian Threshold Image", thresh, True)
# edged = thresh

if DISPLAY_STEP_BY_STEP :
	##### Perform edge detection with variable threshold
	# Create Window
	source_window = "Select threshold"
	cv2.namedWindow(source_window)
	cv2.imshow(source_window, image)
	max_thresh = 255
	thresh = 50 # initial threshold
	cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
	thresh_callback(thresh)

	cv2.waitKey()
	cv2.destroyAllWindows()
else :
	##### Perform edge detection with fix threshold
	edged = cv2.Canny(blur, threshold, threshold*2) # canny(img,min,max) #default was 50
	# show_image("Edged Image", edged, False)
	# Then perform a dilation + erosion to close gaps in between object edges
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	# show_image("Dilate and erode image", edged, True)
	# find contours in the edge map
	g_contours, g_hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
	
print("Total number of contours are: ", len(g_contours))
print("Edge threshold choosen: ", threshold)
##### End threshold and edge detection

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(g_contours, _) = contours.sort_contours(g_contours)
pixelsPerMetric = None

if (DISPLAY_STEP_BY_STEP == False) :
	orig = image.copy() # Uncomment (and comment the other one) to have all objets size in one image
else :
	cv2.imshow("Contour detection", image)
	cv2.waitKey(0)

objectProcessed = 0
# loop over the contours individually
for i in range(len(g_contours)):
	# Get the contour
	c = g_contours[i]
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	##### Calculate dominant color in the contour #####
	mask = np.zeros(image.shape[:2], dtype="uint8") # Get a binary blank mask with the same size as the original image
	cv2.drawContours(mask, g_contours, i, 255, -1, cv2.LINE_8, g_hierarchy, 0)
	# cv2.drawContours(mask, c, -1, 255, -1)		# Draw the contour of the object (filled) to have it's mask
	meanVal = cv2.mean(image, mask=mask)[:3] 		#Get the mean RGB (3 values) in the mask area
	objectMeanColor = tuple(map(int, meanVal))		#Convert the array to a single variable and cast it to int
	if DISPLAY_STEP_BY_STEP :
		masked_image = cv2.bitwise_and(image, image, mask=mask)
		cv2.imshow("mask", mask)
		cv2.imshow("masked", masked_image)

	##### Calculate the object size #####
	# compute the rotated bounding box of the contour
	if DISPLAY_STEP_BY_STEP :
		orig = image.copy() # Uncomment (and comment the other one) to have the size of 1 object at a time
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]
		print("Pixel Per Metric : {} px/cm".format(pixelsPerMetric))
		# Display in the image which object is the standard meter
		blx, bly = bl
		cv2.putText(orig, "Reference", (int(blx), int(bly + REF_DISP_PX_OFFSET)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)
	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	# draw the object sizes on the image
	cv2.putText(orig, "{:.3f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, COLOR_WHITE, 2)
	cv2.putText(orig, "{:.3f}cm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, COLOR_WHITE, 2)
	# Write the distances and mean color in the terminal
	objectProcessed += 1
	print("Objet number {} :".format(objectProcessed))
	print("		dimA : {:.3f} cm".format(dimA))
	print("		dimB : {:.3f} cm".format(dimB))
	print("		Mean color: BGR {}".format(objectMeanColor))

	##### Object identification #####
	lineColor = COLOR_RED # object not recongnized
	if(abs(dimA-dimB) <= CIRCLE_RADIUS_DELTA_CM) :
		radius = (dimA+dimB)/2
		print("		radius : {:.3f} cm".format(radius))
		# Find the index of the closest radius to the calculated radius
		index = np.abs(CONTAINER_RADIUS_ARR_CM - radius).argmin()
		difference = abs(CONTAINER_RADIUS_ARR_CM[index] - radius)
		if(difference <= CONTAINER_RADIUS_DELTA_CM) :
			# Verify the color
			containterColor = CONTAINER_COLOR_ARR_BGR[index]
			# Calculate the euclidian distance between the two color in the color space (3D because BGR)
			colorDistance = np.linalg.norm(np.array(objectMeanColor) - np.array(containterColor))
			if(colorDistance <= CONTAINER_COLOR_TOLERANCE):
				print("		Color match, BGR delta : {:.2}".format(colorDistance))
			else:
				print("		Color doesn't match, BGR delta : {:.2}".format(colorDistance))

			print("		Object n°{} found".format(index))
			lineColor = COLOR_GREEN # object recongnized
			tlx, tly = tl
			cv2.putText(orig, "Radius {:.3f}cm".format(radius),
				(int(tlx), int(tly + OBJECT_DISP_PX_OFFSET_FIRST)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, COLOR_WHITE, 2)
			cv2.putText(orig, "Color: BGR {}".format(objectMeanColor),
				(int(tlx), int(tly + OBJECT_DISP_PX_OFFSET_SECOND)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, COLOR_WHITE, 2)
			cv2.putText(orig, "Container {}".format(index),
				(int(tlx), int(tly + OBJECT_DISP_PX_OFFSET_THIRD)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, COLOR_WHITE, 2)
		else :
			print("		Unknow circle object found")
	
	##### Draw all the contours of the objects #####
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, objectMeanColor, 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, COLOR_MAGENTA, -1)	
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, COLOR_BLUE, -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, COLOR_BLUE, -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, COLOR_BLUE, -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, COLOR_BLUE, -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		lineColor, 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		lineColor, 2)

	# show the output image with the current object
	if DISPLAY_STEP_BY_STEP :
		cv2.imshow("Contour detection", orig)
		cv2.waitKey(0)

print("Total contours processed: ", objectProcessed)
# show the output image with all the objects
if (DISPLAY_STEP_BY_STEP == False) :
	cv2.imshow("Contour detection", orig)
	print("Press any key to exit")
	cv2.waitKey(0)