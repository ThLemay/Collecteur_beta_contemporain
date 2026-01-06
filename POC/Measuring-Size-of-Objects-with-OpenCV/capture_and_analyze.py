#################################
# 			Usage				#
#################################

# python capture_and_analyze.py

#################################
# 			Includes			#
#################################

# import the necessary packages for image analysis
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
# import the camera package
from picamera2 import Picamera2, Preview
import time
from libcamera import controls
# With video output
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
# Without video output
from picamera2.encoders import Encoder #Raw encoder ("null")
from picamera2.outputs import CircularOutput

#################################
# 			Defines				#
#################################

# Camera and output resolution
RESOLUTION_HD  = (1280, 720)
RESOLUTION_qHD = (960, 540)
RESOLUTION_nHD = (640, 360)

RESOLUTION_VIDEO_MAX = (2304, 1296) #max FOV
RESOLUTION_VIDEO_MIN = (1536, 864)

CAMERA_RESOLUTION		= RESOLUTION_VIDEO_MAX # Also define the FOV
PROCESSING_RESOLUTION	= RESOLUTION_qHD  # Reduce to increase analyze speed

# Colors are in BGR format
COLOR_BLUE		= (255, 0  , 0  )
COLOR_GREEN 	= (0  , 255, 0  )
COLOR_RED		= (0  , 0  , 255)
COLOR_MAGENTA 	= (255, 0  , 255)
COLOR_WHITE		= (255, 255, 255)

# Detection parameters
THRESHOLD = 50
FILTER_MIN_AREA_PX = 200 # Don't analyse object found (with contour) with an area below X pixels
CIRCLE_RADIUS_DELTA_CM = 1 # Delta between Width and Length to be considered a circle
CONTAINER_RADIUS_ARR_CM = np.array([12.59, 14.46, 16.41, 18.24, 22.4]) # Set the different raius of the object to find here, the order is important ! (the index is used to tell which object is found)
CONTAINER_RADIUS_DELTA_CM = 1 # Delta beteween the calculated radius and the array to be considered valid
CONTAINER_COLOR_ARR_BGR = np.array([
    [160, 170, 160],    # Container 0 (12.59 cm)
    [160, 170, 160],    # Container 1 (14.46 cm)
    [160, 170, 160],    # Container 2 (16.41 cm)
    [160, 170, 160],  # Container 3 (18.24 cm)
    [35 , 35 , 35]   # Container 4 (22.4 cm)
])
CONTAINER_COLOR_TOLERANCE = 30

# Display object name and radius offsets
OBJECT_DISP_PX_OFFSET_FIRST = 25 # Reference to the Y axis of the bottom left point
OBJECT_DISP_PX_OFFSET_SECOND = OBJECT_DISP_PX_OFFSET_FIRST + 25 # 25 is the size of the text above and a little space
OBJECT_DISP_PX_OFFSET_THIRD = OBJECT_DISP_PX_OFFSET_SECOND + 25

# If recording duration is -1, don't save the recording in .mp4 format and do a continuous recording
RECORDING_DURATION_SEC = -1

# Metrics
WIDTH =  3.872 #size in cm of the reference object # change with mm/px later 
PX_PER_METRIC_DEFAULT = 17.5 #None #28.20 # None #px/cm, must be 'None' if a reference object is used

g_fps = 0

#################################
# 			Functions			#
#################################

# Calculate midpoint
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Show image and wait for key to continue
def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()

def find_contours_and_display(image):
	# load the image, convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(image, (7, 7), 0)

	# ##### adpativeThreshold detection
	# thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,12)
	# show_image("Adaptive Threshold Image", thresh, True)
	# edged = thresh
	##### Perform edge detection with fix threshold
	edged = cv2.Canny(blur, THRESHOLD, THRESHOLD*2) # canny(img,min,max) #default was 50
	# show_image("Edged Image", edged, False)
	# Then perform a dilation + erosion to close gaps in between object edges
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	# show_image("Dilate and erode image", edged, True)
	print("Edge threshold choosen: ", THRESHOLD)
	##### End threshold and edge detection

	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
	cnts = imutils.grab_contours(cnts)
	print("Total number of contours is: ", len(cnts))
	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	if len(cnts) > 0:
		(cnts, _) = contours.sort_contours(cnts)

	# Start the computation of the contours
	pixelsPerMetric = PX_PER_METRIC_DEFAULT
	orig = image.copy()
	#orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	objectProcessed = 0
	# loop over the contours individually
	for i in range(len(cnts)):
		c = cnts[i]
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < FILTER_MIN_AREA_PX:
			continue
		
		##### Calculate dominant color in the contour #####
		mask = np.zeros(image.shape[:2], dtype="uint8") # Get a blank mask with the same size as the original image
		cv2.drawContours(mask, cnts, i, 255, -1) 		# Draw the contour of the object (filled) to have it's mask
		meanVal = cv2.mean(image, mask=mask)[:3] 		#Get the mean RGB (3 values) in the mask area
		objectMeanColor = tuple(map(int, meanVal))		#Convert the array to a single variable and cast it to int
		
# 		masked_image = cv2.bitwise_and(image, image, mask=mask)
# 		cv2.imshow("masked", masked_image)
# 		cv2.waitKey(0)
		##### Calculate the object size #####
		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box with the mean color of the object
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, objectMeanColor, 2)
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
			pixelsPerMetric = dB / WIDTH
			print("Pixel Per Metric : {} px/cm".format(pixelsPerMetric))
			# Display in the image which object is the standard meter
			blx, bly = bl
			cv2.putText(orig, "Standard", (int(blx), int(bly + OBJECT_DISP_PX_OFFSET_FIRST)),
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
		print("		area : {} px".format(cv2.contourArea(c)))
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
				colorDelta = np.linalg.norm(np.array(objectMeanColor) - np.array(containterColor))
				if(colorDelta <= CONTAINER_COLOR_TOLERANCE):
					print("		Color match")
				else:
					print("		Color doesn't match")

				# Display container informations
				print("		Object n°{} found".format(index))
				lineColor = COLOR_GREEN # object recognized
				blx, bly = bl
				cv2.putText(orig, "Container {}".format(index),
					(int(blx), int(bly + OBJECT_DISP_PX_OFFSET_FIRST)), cv2.FONT_HERSHEY_SIMPLEX,
					0.65, COLOR_WHITE, 2)
				cv2.putText(orig, "Radius {:.3f}cm".format(radius),
					(int(blx), int(bly + OBJECT_DISP_PX_OFFSET_SECOND)), cv2.FONT_HERSHEY_SIMPLEX,
					0.65, COLOR_WHITE, 2)
				cv2.putText(orig, "Color: BGR {}".format(objectMeanColor),
					(int(blx), int(bly + OBJECT_DISP_PX_OFFSET_THIRD)), cv2.FONT_HERSHEY_SIMPLEX,
					0.65, COLOR_WHITE, 2)
			else :
				print("		Unknow circle object found")

		##### Draw all the contours of the objects #####
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, lineColor, 2)
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, COLOR_MAGENTA, -1)
		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, COLOR_BLUE, -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, COLOR_BLUE, -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, COLOR_BLUE, -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, COLOR_BLUE, -1)
		# draw lines between the midpoints
		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), objectMeanColor, 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), objectMeanColor, 2)

	##### show the output image #####
	cv2.putText(orig, "fps : {}".format(g_fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)
	cv2.putText(orig, "Press 'q' to exit", (0, PROCESSING_RESOLUTION[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2) # Y picture size minus text size
	cv2.imshow("Contour detection", orig)
	print("Total contours processed: ", objectProcessed)
	return orig


def record_and_display_video(duration):
    global g_fps
    try:
        # If duration is -1, go into continuous acquisition mode
        if duration == -1:
            print("Continous recording...")
            encoder = Encoder() # 7.1.4 : "null" encoder
            #default CircularOutput : 150 frames and no output file
            video_output = CircularOutput(buffersize=150, file=None) # 7.2.3
        else:    
            print("Recording for {} seconds...".format(duration))
            encoder = H264Encoder(bitrate=10000000) # 7.1.1
            video_output = FfmpegOutput("Test_Video.mp4") # 7.2.2  
        # Init video output
        video_output_analyzed = 'analyzed_video.mp4'
        fps = 9 #average fps when processing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoOut = cv2.VideoWriter(video_output_analyzed, fourcc, fps, PROCESSING_RESOLUTION)
        # Start video recording
        camera.start_recording(encoder,output=video_output)
        start_time = time.time()
        iteration = 0
        #Add continuous autofocus
        camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        current_time = time.time()
        #Loop for the given duration, if duration is -1 loop indifinitly
        while (current_time - start_time < duration) or (duration == -1) :
            # Save current image (main feed, not the preview one to display the actual quality)
            frame_bgr = camera.capture_array("main")
            # find and display contours
            print("Analyze image...")
            resized_frame = cv2.resize(frame_bgr, PROCESSING_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            analyzed_frame = find_contours_and_display(resized_frame)
            videoOut.write(analyzed_frame)
            
            # Quit if "q" is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if (time.time() - current_time) >= 1 :
                current_time = time.time()
                print("Current fps : {:2}".format(g_fps))
                g_fps = iteration
                iteration = 0
            else :
                iteration += 1
        # Stop recording
        camera.stop_recording()
    finally:
        camera.stop()  # Stop the camera
        videoOut.release()
        camera.close()  # Free resources
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print("Recording ended and ressources are freed")

#################################
# 			Main				#
#################################

# Specify the camera to use
camera = Picamera2(camera_num=0)
# Create camera the configuration
camera_config = camera.create_video_configuration(
    main={"size": CAMERA_RESOLUTION, "format": "RGB888"},  # Resolutiuon in 16/9 closest to 1536x864p720 (can be adjusted, max is : 4608,2592), and in BGR order (see 4.2.2.2)
    controls={"FrameRate": 30} # Set low FPS because calculation take some times (default is 120 or 60fps)
)
#Configure the camera
camera.configure(camera_config)

# Film and display a video with fix duration but can be exiting with "q"
record_and_display_video(RECORDING_DURATION_SEC)
