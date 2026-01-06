print("Init camera.py...")

#################################
#           Import				#
#################################

import time
from time import sleep
from typing import Tuple
# Import the necessary packages for image analysis
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from math import pi
# Import the camera package
from picamera2 import Picamera2, Preview
# To control focus
from libcamera import controls
# With video output
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
# Without video output
from picamera2.encoders import Encoder #Raw encoder ("null")
from picamera2.outputs import CircularOutput
# Error code output
from enum import Enum
# Save calibration data
import os

#################################
#           Defines				#
#################################

### Camera ###
WINDOW_CAMERA_NAME      = "Contour detection"
# Camera and output resolution
RESOLUTION_HD  = (1280, 720)
RESOLUTION_qHD = (960, 540)
RESOLUTION_nHD = (640, 360)
RESOLUTION_VIDEO_MAX = (2304, 1296) # Max FOV => less distortion
RESOLUTION_VIDEO_MIN = (1536, 864)

# Used resolution
CAMERA_RESOLUTION        = RESOLUTION_VIDEO_MAX # Also define the FOV
PROCESSING_RESOLUTION    = RESOLUTION_qHD       # Reduce to increase analyze speed
fps                      = 10

# Crop values
CROP_WIDTH_PX_LEFT       = 150
CROP_WIDTH_PX_RIGHT      = 200 #250 to not see the gear of the trapdoor
CROP_HEIGHT_PX_TOP       = 25
CROP_HEIGHT_PX_BOTTOM    = 0

# Colors are in BGR format
COLOR_BLUE      = (255, 0  , 0  )
COLOR_GREEN     = (0  , 255, 0  )
COLOR_RED       = (0  , 0  , 255)
COLOR_MAGENTA   = (255, 0  , 255)
COLOR_YELLOW    = (0  , 255, 255)
COLOR_CYAN      = (255, 255, 0  )
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0  , 0  , 0  )

# Detection parameters
THRESHOLD = 25
FILTER_MIN_AREA_PX = 5000 # Don't analyse object found (with contour) with an area below X pixels
CIRCLE_RADIUS_DELTA_CM = 1 # Delta between Width and Length to be considered a circle
CONTAINER_RADIUS_ARR_CM = np.array([12.59, 14.46, 16.41, 18.24, 22.4]) # Set the different raius of the object to find here, the order is important ! (the index is used to tell which object is found)
CONTAINER_RADIUS_DELTA_CM = 1 # Delta beteween the calculated radius and the array to be considered valid
CONTAINER_COLOR_ARR_BGR = np.array([
    [187, 200, 181],    # Container 0 (12.59 cm)
    [198, 210, 185],    # Container 1 (14.46 cm)
    [200, 214, 190],    # Container 2 (16.41 cm)
    [201, 216, 193],    # Container 3 (18.24 cm)
    [152 ,151  ,137]    # Container 4 (22.40 cm)
])
CONTAINER_COLOR_TOLERANCE = 8 # Max vector size between actual color and array value, think of BGR like XYZ
CONTAINER_COLOR_TOLERANCE_ARR = np.array([40, 40, 40, 16, 16]) # Container 0 to 4 25, 20, 20, 15, 10 
CONTAINER_TOP_WIDTH_CM = 0.63
# Display object name and radius offsets
OBJECT_DISP_PX_OFFSET_FIRST = 25 # Reference to the Y axis of the bottom left point
OBJECT_DISP_PX_OFFSET_SECOND = OBJECT_DISP_PX_OFFSET_FIRST + 25 # 25 is the size of the text above and a little space
OBJECT_DISP_PX_OFFSET_THIRD = OBJECT_DISP_PX_OFFSET_SECOND + 25

# Metrics
WIDTH = 3.872 #size in cm of the reference object # change with mm/px later 
PX_PER_METRIC_DEFAULT = 17.41 #17.64 #None #28.20 # None #px/cm, must be 'None' if a reference object is used
PX_PER_METRIC_ARR = np.array([16.99, 17.08, 17.5, 17.87, 17.41]) # Container 0 to 4

# Calibration
CALIBRATION_OUTPUT_FILE = "calibration_data.txt"

class containerErrorCode(Enum):
    VALID = 0
    NOT_RECOGNIZED = -1
    MISSMATCH_SIZE = -2
    MISSMATCH_COLOR = -3
    
class containerInfo_t:
    def __init__(self, id, dimPx, dimCm, color, colorDelta, error):
        self.id = id
        self.dimPx = dimPx
        self.dimCm = dimCm
        self.color = color
        self.colorDelta = colorDelta
        self.error = error

#################################
#		Private Variables		#
#################################

### Waiting list ###
s_waitingOrderArr = [0]
s_waitingIndex = 0

#################################
#		Private Functions		#
#################################
# Private functions ( "_" as a prefix)

# Calculate midpoint
def _midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Show image and wait for key to continue
def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(1)
    if destroy_all:
        cv2.destroyAllWindows()
    
def nothing(x):
   pass
def show_canny_preview(grayBlurImg):
    cv2.namedWindow("Canny Thresholds")
    cv2.createTrackbar("Lower", "Canny Thresholds", s_lowThresh, 255, nothing)
    cv2.createTrackbar("Upper", "Canny Thresholds", s_highThresh, 255, nothing)
    cv2.createTrackbar("LockRatio 2x", "Canny Thresholds", 0, 1, nothing)

    while True:
        lower = cv2.getTrackbarPos("Lower", "Canny Thresholds")
        lock = cv2.getTrackbarPos("LockRatio 2x", "Canny Thresholds")

        if lock:
            upper = min(2 * lower, 255)
            cv2.setTrackbarPos("Upper", "Canny Thresholds", upper)
        else:
            upper = cv2.getTrackbarPos("Upper", "Canny Thresholds")
            
        edges = cv2.Canny(grayBlurImg, lower, upper)
        cv2.imshow("Canny Thresholds", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow("Canny Thresholds")
    return lower, upper

s_lowThresh = THRESHOLD
s_highThresh = THRESHOLD*2

def find_contours(image, calibrationModeEn = False):
    # findContours work with BGR image, no need to convert it to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray must be used for adaptive threshold, but not needed for Canny
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #show_image("Blur gray", blur, False)

    """
    ##### adpativeThreshold detection
    # show_image("Blur Gaussian image", blur, False)
    blockSize   = 11 #5
    constC      = 3  #1
    thresh = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,blockSize,constC)
    # show_image("Adaptive Threshold Image", thresh, False)
    
    
    """
    ##### Perform edge detection with fix threshold
    """
    # Display the canny detection 
    global s_lowThresh, s_highThresh
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # Active slider mode
        s_lowThresh, s_highThresh = show_canny_preview(blur)
    """
    
    print("Low thresh = {} | High Thresh = {}".format(s_lowThresh, s_highThresh))
    edged = cv2.Canny(blur, s_lowThresh, s_highThresh)
    #edged = cv2.Canny(blur, THRESHOLD, THRESHOLD*2) # canny(img,min,max) #default was 50
    #show_image("Edged Image", edged, False)
    
    
    # Choose the detection to choose :
    #   - thresh for adaptive
    #   - edged for fix
    #edged = thresh
    
    # Then perform a dilation + erosion to close gaps in between object edges
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    #show_image("Dilate and erode image", edged, False)
    """
    # Does the same thing:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # show_image("Clean Threshold Image", thresh_clean, False)
    edged = thresh_clean
    """
    
    ##### End threshold and edge detection
    
    # find contours in the edge map
    # Only the most external contour and with simple point (lighter and don't need a more detailed contour)
    #RETR_EXTERNAL
    #RETR_TREE
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
    print("Total number of contours is: ", len(cnts))
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    if len(cnts) > 0:
        (cnts, _) = contours.sort_contours(cnts)
        
    # Start the computation of the contours   
    if(calibrationModeEn == False):
        pixelsPerMetric = PX_PER_METRIC_ARR[s_waitingOrderArr[s_waitingIndex]]
    else:
        pixelsPerMetric = PX_PER_METRIC_DEFAULT
    print("Pixel Per Metric from container n°{} : {} px/cm".format(s_waitingOrderArr[s_waitingIndex], pixelsPerMetric))
    orig = image.copy()
    #orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    objectProcessed = 0
    containerId = -1
    containerDimPx = (0, 0) # X Y
    containerDimCm = (0.0, 0.0) # X Y
    containerColor = (0, 0, 0) # BGR
    containerInfo  = (0, 0, 0.0, 0.0, 0, 0, 0) # DimPx + DimCm + Color
    containerError = containerErrorCode.NOT_RECOGNIZED
    containerColorDetla = 0
    # loop over the contours individually
    for i in range(len(cnts)):
        c = cnts[i]
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < FILTER_MIN_AREA_PX:
            continue
        
        ##### Calculate the object size #####
        # compute the minimum enclosing circle
        (center_x, center_y), radius_px = cv2.minEnclosingCircle(c)
        center = (int(center_x), int(center_y))
        radius_px = int(radius_px)
        
        area_contour = cv2.contourArea(c)
        area_circle = pi * radius_px**2

        ratio = area_contour / area_circle
        if(ratio < 0.90):
            #print("Contour is abnormal, ignore it (ratio = {:.2f})".format(ratio))
            continue
            
        h = hierarchy[i]
        print(f"Contour {i} => Next: {h[0]}, Prev: {h[1]}, First_Child: {h[2]}, Parent: {h[3]}")
        
        cv2.circle(orig, center, radius_px, (0, 255, 0), 2)
        cv2.putText(orig, "{:.2f}px | Ratio: {:.2f}".format(radius_px, ratio),
            (int(center_x)-10, int(center_y)-int(radius_px)-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, COLOR_WHITE, 2)
            
        # For the masks
        OuterCircleWidth = int(CONTAINER_TOP_WIDTH_CM * pixelsPerMetric)
        radiusInnerCircle = radius_px - OuterCircleWidth
        radiusBottomContainer = int(0.65* radius_px)
        
        ##### Calculate dominant color in the contour #####
        mask = np.zeros(image.shape[:2], dtype="uint8") # Get a blank mask with the same size as the original image
        
        ### Outer edge image color
        maskInnerCircle = mask.copy()
        # Draw the contour of the object (filled) to have it's mask
        cv2.drawContours(maskInnerCircle, cnts, i, 255, -1)         
        # Substract the inner circle to only have the outer circle for the color
        cv2.circle(maskInnerCircle, center, radiusInnerCircle, 0, -1)
        meanValOuter = cv2.mean(image, mask=maskInnerCircle)[:3]                     #Get the mean RGB (3 values) in the mask area
        objectMeanColorOuter = tuple(map(int, meanValOuter))                         #Convert the array to a single variable and cast it to int
        masked_image = cv2.bitwise_and(image, image, mask=maskInnerCircle)
        #show_image("masked container color", masked_image, False)
        
        ### Content image color
        maskBottom= mask.copy()
        cv2.circle(maskBottom, center, radiusBottomContainer, 255, -1)
        meanValBottom = cv2.mean(image, mask=maskBottom)[:3]                          #Get the mean RGB (3 values) in the mask area
        objectMeanColorBottom = tuple(map(int, meanValBottom))                        #Convert the array to a single variable and cast it to int
        masked_image = cv2.bitwise_and(image, image, mask=maskBottom)
        #show_image("masked container content", masked_image, False)
        
        objectMeanColor = objectMeanColorBottom
        
        ##### compute the rotated bounding box of the contour #####
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
        (tltrX, tltrY) = _midpoint(tl, tr)
        (blbrX, blbrY) = _midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = _midpoint(tl, bl)
        (trbrX, trbrY) = _midpoint(tr, br)
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
        cv2.putText(orig, "{:.2f}cm".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, COLOR_WHITE, 2)
        cv2.putText(orig, "{:.2f}cm".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, COLOR_WHITE, 2)
        # Write the distances and mean color in the terminal
        objectProcessed += 1
        print("Objet number {} :".format(objectProcessed))
        print("        ratio: {:.2f}".format(ratio))
        print("        area : {} px".format(cv2.contourArea(c)))
        print("        dimA : {:.2f} cm".format(dimA))
        print("        dimB : {:.2f} cm".format(dimB))
        print("        Mean color: BGR {}".format(objectMeanColor))
        
        
        ##### Object identification #####
        lineColor = COLOR_RED # object not recongnized
        if(abs(dimA-dimB) <= CIRCLE_RADIUS_DELTA_CM) :
            radius = (dimA+dimB)/2
            print("        radius calculated : {:.2f} cm".format(radius))
            print("        diameter contour : {:.2f} cm".format(2*radius_px/pixelsPerMetric))
            # Find the index of the closest radius to the calculated radius
            index = np.abs(CONTAINER_RADIUS_ARR_CM - radius).argmin()
            containerId = index
            difference = abs(CONTAINER_RADIUS_ARR_CM[index] - radius)
            if((difference <= CONTAINER_RADIUS_DELTA_CM) or (calibrationModeEn == True)) :
                # Verify the color
                containterColor = CONTAINER_COLOR_ARR_BGR[index]
                colorDelta = np.linalg.norm(np.array(objectMeanColor) - np.array(containterColor))
                containerColorDetla = colorDelta
                print("        Color tolerance = {}".format(CONTAINER_COLOR_TOLERANCE_ARR[index]))
                if((colorDelta <= CONTAINER_COLOR_TOLERANCE_ARR[index]) or (calibrationModeEn == True)):
                    print("        Color match")
                    lineColor = COLOR_GREEN # object recognized
                    containerError = containerErrorCode.VALID
                else:
                    print("        Color doesn't match, diffrence is {}".format(colorDelta))
                    containerError = containerErrorCode.MISSMATCH_COLOR
                
                # Display container informations
                print("        Object n°{} found".format(index))
                # Get container informations
                containerDimPx = dA, dB
                containerDimCm = dimA, dimB
                containerColor = objectMeanColor
                
                blx, bly = bl
                cv2.putText(orig, "Container {}".format(index),
                    (int(blx), int(bly + OBJECT_DISP_PX_OFFSET_FIRST)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, COLOR_WHITE, 2)
                cv2.putText(orig, "Radius {:.2f}cm".format(radius),
                    (int(blx), int(bly + OBJECT_DISP_PX_OFFSET_SECOND)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, COLOR_WHITE, 2)
                cv2.putText(orig, "Color: BGR {}".format(objectMeanColor),
                    (int(blx), int(bly + OBJECT_DISP_PX_OFFSET_THIRD)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, COLOR_WHITE, 2)
                cv2.putText(orig, "{:.2f}px | Ratio: {:.2f}".format(radius_px, ratio),
                    (int(center_x)-10, int(center_y)-int(radius_px)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, COLOR_WHITE, 2)
            else :
                print("        Unknow circle object found")
                containerError = containerErrorCode.MISSMATCH_SIZE

        ##### Draw all the contours of the objects #####
        cv2.circle(orig, center, radius_px, lineColor, 2)

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
        
        # Only take the first "correct" (circular) contour found
        # Comment to display all "correct" contour
        break

    print("Total contours processed: ", objectProcessed)
    
    """
    # Test values
    containerId = 1
    containerDimPx = (50,50)
    containerDimCm = (12.25,12.30)
    containerColor = (220, 234, 219)
    containerColorDetla = 15.14
    containerError = containerErrorCode.MISSMATCH_COLOR
    """
    
    containerInfoRet = containerInfo_t(
        id=containerId,
        dimPx=containerDimPx,
        dimCm=containerDimCm,
        color=containerColor,
        colorDelta=containerColorDetla,
        error=containerError
    )
    
    return orig, containerInfoRet

def calculate_container_avg(measures):
    moyXpx = sum([m[0][0] for m in measures]) / len(measures)
    moyYpx = sum([m[0][1] for m in measures]) / len(measures)
    moyXcm = sum([m[1][0] for m in measures]) / len(measures)
    moyYcm = sum([m[1][1] for m in measures]) / len(measures)
    moyB   = sum([m[2][0] for m in measures]) / len(measures)
    moyG   = sum([m[2][1] for m in measures]) / len(measures)
    moyR   = sum([m[2][2] for m in measures]) / len(measures)
    return (moyXpx, moyYpx), (moyXcm, moyYcm), (moyB, moyG, moyR)
    
def calibrate_disp(pos):
    posX = 0
    posY = 0
    circleRadiusPx = 20
    if pos == 0:
        #Reset values
        calibrate_disp.image = np.zeros((300,300,3), dtype="uint8")
        calibrate_disp.oldPosX = 0
        calibrate_disp.oldPosY = 0
    #Turn old circle to green if it exist
    if(calibrate_disp.oldPosX != 0) and (calibrate_disp.oldPosY !=0) :
        cv2.circle(calibrate_disp.image,
                   (int(calibrate_disp.oldPosX), int(calibrate_disp.oldPosY)),
                   circleRadiusPx, COLOR_GREEN, -1)
    match pos:
        case 0: # Bottom right
            posX = calibrate_disp.image.shape[1] - circleRadiusPx
            posY = calibrate_disp.image.shape[0] - circleRadiusPx
        case 1: # Top right
            posX = calibrate_disp.image.shape[1] - circleRadiusPx
            posY = circleRadiusPx
        case 2: # Top left
            posX = circleRadiusPx
            posY = circleRadiusPx
        case 3: # Bottom left
            posX = circleRadiusPx
            posY = calibrate_disp.image.shape[0] - circleRadiusPx
        case 4: # Center
            posX = calibrate_disp.image.shape[1]/2
            posY = calibrate_disp.image.shape[0]/2
        
        case _:
            print(f"Invalid position : {pos}")
            return 
    
    cv2.circle(calibrate_disp.image, (int(posX), int(posY)), circleRadiusPx, COLOR_YELLOW, -1)
    cv2.imshow("Calibration POS", calibrate_disp.image)
    cv2.moveWindow("Calibration POS", 100, 300)
    calibrate_disp.oldPosX = posX
    calibrate_disp.oldPosY = posY

def calibrate_compute(id, measurments):
    # Exit if no container detected
    if(id == -1): return
    
    calibrate_compute.measure.append(measurments)
    lenArrMeas = len(calibrate_compute.measure)
    print(f"Measure {lenArrMeas} added : Xpx,Ypx,Xcm,Ycm,B,G,R = {measurments}")
    
    if lenArrMeas >= 5:
        calibrate_disp(0)
    else:
        calibrate_disp(lenArrMeas)
        
    if lenArrMeas == 5 :
        # Get the average of all values
        dimPxMoy, dimCmMoy, colorMoy = calculate_container_avg(calibrate_compute.measure)
        # Write the values in a text file
        fileName = CALIBRATION_OUTPUT_FILE
        # Check if the file exists and is not empty
        fileExists = (os.path.isfile(fileName)) and (os.stat(fileName).st_size > 0)
        with open(fileName, "a") as file:
            # write header if it doesn't exist
            if not fileExists :
                 file.write("id;x_px;y_px;x_cm;y_cm;color_b;color_g;color_r\n")
           # Write the averages to the file
            file.write(f"{id};{int(dimPxMoy[0])};{int(dimPxMoy[1])};{dimCmMoy[0]:.2f};{dimCmMoy[1]:.2f};{int(colorMoy[0])};{int(colorMoy[1])};{int(colorMoy[2])}\n")
        print(f"Average written to {fileName} with id = {id} : size = {dimPxMoy}px / {dimCmMoy}cm, color = {colorMoy}")
                
        # Reset values
        calibrate_compute.measure = []

def get_crop_ratio():
    return cropRatio
   
def get_frame():
    global camera
    return camera.capture_array("main")

def resize_and_crop_frame(frameToCrop):
    resizedFrame = cv2.resize(frameToCrop, PROCESSING_RESOLUTION, interpolation=cv2.INTER_LINEAR)
    return resizedFrame[CROP_HEIGHT_PX_TOP:cropHeight, CROP_WIDTH_PX_LEFT:cropWidth]

def get_and_analyze_frame(CalibrationModeEn = False):
    return find_contours(resize_and_crop_frame(get_frame()), CalibrationModeEn)
        
def uninit():
    global camera
    print("[Camera] Uninit the peripheral...")
    camera.stop()  # Stop the camera
    camera.close()  # Free resources
    print("[Camera] Uninit done")

#################################
#		Initialization			#
#################################

### Crop captured image ###
# Process the cropped frame size and the display size
# Cropped size :
width, height = PROCESSING_RESOLUTION
cropWidth = width-CROP_WIDTH_PX_RIGHT
cropHeight = height-CROP_HEIGHT_PX_BOTTOM
# Displayed size :
width = cropWidth-CROP_WIDTH_PX_LEFT
height = cropHeight-CROP_HEIGHT_PX_TOP
print(f"cropWidth = {width} | cropHeight = {height}")
cropRatio = width/height

### Camera ###
# Specify the camera to use
camera = Picamera2(camera_num=0)
# Create camera the configuration
camera_config = camera.create_video_configuration(
    main={"size": CAMERA_RESOLUTION, "format": "RGB888"},  # Resolutiuon in 16/9 closest to 1536x864p720 (can be adjusted, max is : 4608,2592), and in BGR order (see 4.2.2.2)
    controls={"FrameRate": 30} # Set low FPS because calculation take some times (default is 120 or 60fps)
)
# Configure the camera
camera.configure(camera_config)
# Configure encoder for continuous recording
encoder = Encoder() # 7.1.4 : "null" encoder
# default CircularOutput : 150 frames and no output file
video_output = CircularOutput(buffersize=150, file=None) # 7.2.3
# Start video recording
camera.start_recording(encoder,output=video_output)
# Add continuous autofocus
# camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
# Init static variable in functions
calibrate_compute.measure = []
calibrate_disp.image = np.zeros((300,300,3), dtype="uint8") #3 is for the RGB (1 equal gray scale)
calibrate_disp.oldPosX = 0
calibrate_disp.oldPosY = 0

print("Init camera.py done")

if __name__ == "__main__":
	try:
		print("Test program for camera.py...")
		while True:
			# Capture the image
			bgrFrame = get_frame()
			cropedFrame = resize_and_crop_frame(bgrFrame)
			# Analyze the image
			analyzedFrame, containerInfo = find_contours(cropedFrame)
            #print("ContainerId = {}".format(containerId))
			#show_image("Analyzed Frame", analyzedFrame, False)
			cv2.imshow("Analyzed Frame", analyzedFrame)
			# Get user inpout key
			key = cv2.waitKey(1) & 0xFF
			# Quit if "q" orc ESC key is pressed
			if (key == ord('q')) or (key == 27):
				break
	
	except KeyboardInterrupt:
		print("===== Program interrupted =====")
		   
	finally:
		print("Cleaning the peripherals...")
		uninit()
		cv2.destroyAllWindows()  # Close all OpenCV windows
		# gpio cleanup is automaticly done by gpiozero when script end
		print("===== Program ended ======")
