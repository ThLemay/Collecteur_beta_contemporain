#################################
#             Usage                #
#################################

# Normal use case, detect and dump container
# python scan_and_dump.py

# Calibration of the camera for px/cm ratio and BGR
# python scan_and_dump.py -c
# python scan_and_dump.py --calibration

#################################
#           Includes            #
#################################
import time
from time import sleep
# For script arguments
import argparse 
# For files statistics
import os
### GPIO ###
from gpiozero import Button
### Servo ###
# from adafruit_servokit import ServoKit
### Camera ###
# Import the necessary packages for image analysis
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
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

#################################
#           Defines             #
#################################
### Windows
WINDOW_SCREEN_NAME           = "0117-NUT"
WINDOW_BORDER_X_OFFSET       = 0
WINDOW_OPENCV_TOP_Y_OFFSET   = 50
WINDOW_OPENCV_BOT_Y_OFFSET   = 20
WINDOW_TASK_BAR_Y_OFFSET     = 62
WINDOW_TERMINAL_X_OFFSET     = 1300- WINDOW_BORDER_X_OFFSET #1920x1080
WINDOW_INFO_NAME             = "Container informations"
### Screen ####
# Images name to display
SCREEN_NAME_WELCOME     = "screens/Welcome_page.jpg"
SCREEN_NAME_ACCEPTED    = "screens/Container_accepted.jpg"
SCREEN_NAME_REJECTED    = "screens/Container_rejected.jpg"
SCREEN_NAME_XS          = "screens/Container_XS.jpg"
SCREEN_NAME_S           = "screens/Container_S.jpg"
SCREEN_NAME_M           = "screens/Container_M.jpg"
SCREEN_NAME_L           = "screens/Container_L.jpg"
SCREEN_NAME_XL          = "screens/Container_XL.jpg"
# Make a dictionnary of the container id to their name
CONTAINER_ID_TO_STRING  = {-1:"Non detecte", 0:"XS", 1:"S", 2:"M", 3:"L", 4:"XL"}
### Waiting list ###
FILE_OBJECT_ORDER_NAME = "awaiting_object_list.txt"
WAITING_VALIDATION_DELAY_S = 1.5 # container need to be seen during this period to be validated and dump

### GPIO ###
SWITCH_STANDBY_PIN  = 17 # Limit switch standby
SWITCH_OPEN_PIN     = 18 # Limit switch open

### Servo ###
SERVO_CHANNEL           = 0
DELAY_BEFORE_RETURN_SEC = 2 # is a float, can be 2.5s

### Camera ###
WINDOW_CAMERA_NAME      = "Contour detection"
# Camera and output resolution
RESOLUTION_HD  = (1280, 720)
RESOLUTION_qHD = (960, 540)
RESOLUTION_nHD = (640, 360)
RESOLUTION_VIDEO_MAX = (2304, 1296) #max FOV
RESOLUTION_VIDEO_MIN = (1536, 864)

# Used resolution
CAMERA_RESOLUTION        = RESOLUTION_VIDEO_MAX # Also define the FOV
PROCESSING_RESOLUTION    = RESOLUTION_qHD  # Reduce to increase analyze speed

# Crop values
CROP_WIDTH_PX_LEFT       = 250
CROP_WIDTH_PX_RIGHT      = 270
CROP_HEIGHT_PX_TOP       = 85
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
THRESHOLD = 50
FILTER_MIN_AREA_PX = 100 # Don't analyse object found (with contour) with an area below X pixels
CIRCLE_RADIUS_DELTA_CM = 1 # Delta between Width and Length to be considered a circle
CONTAINER_RADIUS_ARR_CM = np.array([12.59, 14.46, 16.41, 18.24, 22.4]) # Set the different raius of the object to find here, the order is important ! (the index is used to tell which object is found)
CONTAINER_RADIUS_DELTA_CM = 1 # Delta beteween the calculated radius and the array to be considered valid
CONTAINER_COLOR_ARR_BGR = np.array([
    [238, 244, 231],    # Container 0 (12.59 cm)
    [222, 237, 217],    # Container 1 (14.46 cm)
    [204, 227, 206],    # Container 2 (16.41 cm)
    [189, 213, 196],  # Container 3 (18.24 cm)
    [102 ,99  ,93]   # Container 4 (22.4 cm)
])
CONTAINER_COLOR_TOLERANCE = 8 # Max vector size between actual color and array value, think of BGR like XYZ
CONTAINER_COLOR_TOLERANCE_ARR = np.array([10, 5, 6, 5, 10]) # Container 0 to 4
# Display object name and radius offsets
OBJECT_DISP_PX_OFFSET_FIRST = 25 # Reference to the Y axis of the bottom left point
OBJECT_DISP_PX_OFFSET_SECOND = OBJECT_DISP_PX_OFFSET_FIRST + 25 # 25 is the size of the text above and a little space
OBJECT_DISP_PX_OFFSET_THIRD = OBJECT_DISP_PX_OFFSET_SECOND + 25

# Metrics
WIDTH = 3.872 #size in cm of the reference object # change with mm/px later 
PX_PER_METRIC_DEFAULT = 17.64 #None #28.20 # None #px/cm, must be 'None' if a reference object is used
PX_PER_METRIC_ARR = np.array([16.96, 17.25, 17.61, 18.09, 16.99]) # Container 0 to 4
16
# Calibration
CALIBRATION_OUTPUT_FILE = "calibration_data.txt"

class containerErrorCode(Enum):
    VALID = 0
    NOT_RECOGNIZED = -1
    MISSMATCH_SIZE = -2
    MISSMATCH_COLOR = -3

#################################
#           Variables           #
#################################
s_calibrationModeEn = False

### Waiting list ###
s_waitingOrderArr = [0]
s_waitingIndex = 0

### GPIO ###
s_switchStandbyTriggered = False
s_switchOpenTriggered = False

### Servo ###
s_servoStandbyAngle = 90; # Value where we are sure neither of the switches will be triggered

### Camera ###

#################################
#           Functions           #
#################################
### Screen ###
def screen_waiting_container_disp(id): 
    match id:
        case 0: 
            screen = cv2.imread(SCREEN_NAME_XS)
        case 1:
            screen = cv2.imread(SCREEN_NAME_S)
        case 2:
            screen = cv2.imread(SCREEN_NAME_M)
        case 3:
            screen = cv2.imread(SCREEN_NAME_L)
        case 4:
            screen = cv2.imread(SCREEN_NAME_XL)
        case _:
            print(f"Unknow container ID : {id}")
            return 
    cv2.imshow(WINDOW_SCREEN_NAME, screen)
    
def screen_accepted():
    screen = cv2.imread(SCREEN_NAME_ACCEPTED)
    cv2.imshow(WINDOW_SCREEN_NAME, screen)
    
def screen_rejected():
    screen = cv2.imread(SCREEN_NAME_REJECTED)
    cv2.imshow(WINDOW_SCREEN_NAME, screen)

### Waiting list ###
def waiting_list_read_file(filePath):
    objectsOrder = []
    with open(filePath, 'r') as f:
       objectsOrder = [int(ligne.strip()) for ligne in f.readlines()]
    return objectsOrder 

# def waiting_display_id(id, color):
#     fontFace = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 10
#     thickness = 5
#     # Erase previous text (quicker than erasing the whole image)
#     cv2.putText(waiting_display_id.image, waiting_display_id.text, 
#                 (waiting_display_id.textX, waiting_display_id.textY), 
#                 fontFace, fontScale, COLOR_BLACK, thickness)
#     # Create text
#     waiting_display_id.text = str(id)
#     # Get boundary of this text
#     waiting_display_id.textsize = cv2.getTextSize(waiting_display_id.text, 
#                                                   fontFace, fontScale, thickness)[0]
#     # Get coords based on boundary
#     waiting_display_id.textX = int((waiting_display_id.image.shape[1] - waiting_display_id.textsize[0]) / 2)
#     waiting_display_id.textY = int((waiting_display_id.image.shape[0] + waiting_display_id.textsize[1]) / 2)
#     # add text centered on image
#     cv2.putText(waiting_display_id.image, waiting_display_id.text, 
#                 (waiting_display_id.textX, waiting_display_id.textY), 
#                 fontFace, fontScale, color, thickness)
#     # display image
#     print("Waiting display : {}, in BGR : {}".format(id, color))
#     cv2.imshow('Waiting container', waiting_display_id.image)
    
def waiting_process_id(containerId, containerError):
    global s_waitingIndex
    if(containerId != -1): # A known object is found
        #Go back to yellow if no object are found
        waiting_process_id.updateDisp = True
        if((containerId == s_waitingOrderArr[s_waitingIndex]) and (containerError == containerErrorCode.VALID)):
            # First time 
            if(waiting_process_id.detectionTime == 0):
                waiting_process_id.detectionTime = time.time()
                print("     Correct input container n°{}".format(containerId))
                screen_waiting_container_disp(s_waitingOrderArr[s_waitingIndex])
                # waiting_display_id(s_waitingOrderArr[s_waitingIndex], COLOR_GREEN)
                # update display
                # cv2.waitKey(1)
            # Container need to be seen WAITING_VALIDATION_DELAY_S seconds to be validated and dump
            if (time.time() - waiting_process_id.detectionTime) >= WAITING_VALIDATION_DELAY_S :
                screen_accepted()
                # update display
                cv2.waitKey(1)
                # Dump container
                servo_dump_container()
                s_waitingIndex+=1
                if s_waitingIndex == len(s_waitingOrderArr) : 
                    s_waitingIndex = 0
                    print("     Loop waiting list!")
        else:
            # waiting_display_id(s_waitingOrderArr[s_waitingIndex], COLOR_RED)
            screen_rejected()
            print("Wrong input container ! n°{} instead of n°{}".format(containerId, s_waitingOrderArr[s_waitingIndex]))  
            # Reset the detection duration
            waiting_process_id.detectionTime = 0          
    else:
        # Do once, put current or new container ID in yellow
        if(waiting_process_id.updateDisp == True) :
            waiting_process_id.updateDisp = False
            # waiting_display_id(s_waitingOrderArr[s_waitingIndex], COLOR_YELLOW)
            screen_waiting_container_disp(s_waitingOrderArr[s_waitingIndex])
            # Reset the detection duration
            waiting_process_id.detectionTime = 0 

### GPIO ###
# Callback for the gpio
def gpio_switch_standby_cb():
    global s_switchStandbyTriggered
    s_switchStandbyTriggered = True
    print("Stanby switch triggered")
    
def gpio_switch_open_cb():
    global s_switchOpenTriggered
    s_switchOpenTriggered = True
    print("Open switch triggered")
    
### Servo ###
def servo_init_position():
    # try to find standby angle
    currentAngle = kit.servo[0].angle
    if(currentAngle == None):
        # This case happen the first time the servo is used
        print("Warning : servo read angle is null !")
        # Move the servo to a postion between the 2 switch
        currentAngle = 90
        kit.servo[0].angle = currentAngle
        # Let time to move to the desired angle
        sleep(0.5)
        
    while(s_switchStandby.is_pressed == False):
        currentAngle -= 0.5
        kit.servo[0].angle = currentAngle
        sleep(0.05)
    return kit.servo[0].angle

def servo_dump_container():
    print("Dumping container into the bin...")
    global s_switchStandbyTriggered, s_switchOpenTriggered
    s_switchStandbyTriggered = False
    s_switchOpenTriggered = False
    # Move the container
    print("Turning until switch Open is trigger")
    currentAngle = kit.servo[SERVO_CHANNEL].angle
    while not s_switchOpenTriggered:
        kit.servo[SERVO_CHANNEL].angle = currentAngle
        currentAngle += 0.5
        sleep(0.01)
    print("Open angle is : {:.2f}°".format(currentAngle))
    # Stop servo
    print("Stopping servo")
    # Wait for the container to fall
    sleep(DELAY_BEFORE_RETURN_SEC)
    # Return to inital position
    print("Turning until switch Standby is triggered or angle is reached")
    while (not s_switchStandbyTriggered) and (currentAngle > s_servoStandbyAngle) :
        kit.servo[SERVO_CHANNEL].angle = currentAngle
        currentAngle -= 0.5
        sleep(0.005)
    if(not s_switchStandbyTriggered): print("Standby angle reached but switch not triggered")
    print("Standby angle is : {:.2f}°".format(currentAngle))
    print("Dumping container done")

### Camera ###
# Calculate midpoint
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Show image and wait for key to continue
def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()

def camera_find_contours(image):
    # findContours work with BGR image, no need to convert it to gray scale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Blur slightly the image
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
#     print("Edge threshold choosen: ", THRESHOLD)
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
    if(s_calibrationModeEn == False):
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
        
        ##### Calculate dominant color in the contour #####
        mask = np.zeros(image.shape[:2], dtype="uint8") # Get a blank mask with the same size as the original image
        cv2.drawContours(mask, cnts, i, 255, -1)         # Draw the contour of the object (filled) to have it's mask
        meanVal = cv2.mean(image, mask=mask)[:3]         #Get the mean RGB (3 values) in the mask area
        objectMeanColor = tuple(map(int, meanVal))        #Convert the array to a single variable and cast it to int
        
#         masked_image = cv2.bitwise_and(image, image, mask=mask)
#         cv2.imshow("masked", masked_image)
#         cv2.waitKey(0)
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
        cv2.putText(orig, "{:.2f}cm".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, COLOR_WHITE, 2)
        cv2.putText(orig, "{:.2f}cm".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, COLOR_WHITE, 2)
        # Write the distances and mean color in the terminal
        objectProcessed += 1
        print("Objet number {} :".format(objectProcessed))
        print("        area : {} px".format(cv2.contourArea(c)))
        print("        dimA : {:.2f} cm".format(dimA))
        print("        dimB : {:.2f} cm".format(dimB))
        print("        Mean color: BGR {}".format(objectMeanColor))
        
        ##### Object identification #####
        lineColor = COLOR_RED # object not recongnized
        if(abs(dimA-dimB) <= CIRCLE_RADIUS_DELTA_CM) :
            radius = (dimA+dimB)/2
            print("        radius : {:.2f} cm".format(radius))
            # Find the index of the closest radius to the calculated radius
            index = np.abs(CONTAINER_RADIUS_ARR_CM - radius).argmin()
            containerId = index
            difference = abs(CONTAINER_RADIUS_ARR_CM[index] - radius)
            if((difference <= CONTAINER_RADIUS_DELTA_CM) or (s_calibrationModeEn == True)) :
                # Verify the color
                containterColor = CONTAINER_COLOR_ARR_BGR[index]
                colorDelta = np.linalg.norm(np.array(objectMeanColor) - np.array(containterColor))
                containerColorDetla = colorDelta
                print("        Color tolerance = {}".format(CONTAINER_COLOR_TOLERANCE_ARR[index]))
                if((colorDelta <= CONTAINER_COLOR_TOLERANCE_ARR[index]) or (s_calibrationModeEn == True)):
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
            else :
                print("        Unknow circle object found")
                containerError = containerErrorCode.MISSMATCH_SIZE

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

    print("Total contours processed: ", objectProcessed)
    
#     # Test values
#     containerId = 1
#     containerDimPx = (50,50)
#     containerDimCm = (12.25,12.30)
#     containerColor = (220, 234, 219)
#     containerColorDetla = 15.14
#     containerError = containerErrorCode.MISSMATCH_COLOR
    
    containerInfo = containerDimPx, containerDimCm, containerColor
    camera_display_raw_info(containerId, containerDimCm, containerColor, containerColorDetla, containerError)
    return orig, containerId, containerInfo, containerError

def camera_calculate_fps():
    # Calculate fps
    if (time.time() - camera_calculate_fps.frameTime) >= 1 :
        # 1 second elapsed, count the frame that 
        camera_calculate_fps.frameTime = time.time()
        camera_calculate_fps.fps = camera_calculate_fps.framNb
        camera_calculate_fps.framNb = 0
        print("Current fps : {:2}".format(camera_calculate_fps.fps))
    else :
        camera_calculate_fps.framNb += 1

def camera_add_text_and_display(image):
    camera_calculate_fps()
    # Add text
    cv2.putText(image, "fps : {}".format(camera_calculate_fps.fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)
    cv2.putText(image, "Press 'q' to exit", (0, image.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2) # Y picture size minus text size
    if(s_calibrationModeEn) :
        cv2.putText(image, "Press 'c' to take a measurments", (0, image.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)
    # Display
    cv2.imshow(WINDOW_CAMERA_NAME, image)
    return image

def camera_display_raw_info(id, dimCm, color, colorDelta, err):
    global infoFrame
    infoFrame = np.ones(infoFrame.shape, dtype=np.uint8) * 255
    xOffset = 30
    fontScale = 0.9
    if(err == containerErrorCode.NOT_RECOGNIZED) :
        cv2.putText(infoFrame, f"Contenant : {CONTAINER_ID_TO_STRING.get(id)}", (0,xOffset), cv2.FONT_ITALIC, fontScale, COLOR_RED, 2)
    else :
        cv2.putText(infoFrame, f"Contenant : {CONTAINER_ID_TO_STRING.get(id)}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    xOffset+= (25 + 2*5) * fontScale # Text size + 5 top and 5 bottom
    radius = (dimCm[0] + dimCm[1]) / 2
    if(err == containerErrorCode.MISSMATCH_SIZE) :
        cv2.putText(infoFrame, f"Dimension (cm) : {radius:.2f}", (0,int(xOffset)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_RED, 2)
    else :
        cv2.putText(infoFrame, f"Dimension (cm) : {radius:.2f}", (0,int(xOffset)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    xOffset+= (25 + 2*5) * fontScale # Text size + 5 top and 5 bottom
    cv2.putText(infoFrame, f"Couleur (format BVR) : {color}", (0,int(xOffset)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    xOffset+= (25 + 2*5) * fontScale # Text size + 5 top and 5 bottom
    if(err == containerErrorCode.MISSMATCH_COLOR) :
        cv2.putText(infoFrame, f"Coefficient de salissure : {colorDelta:.1f}", (0,int(xOffset)), cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC, fontScale, COLOR_RED, 2)
    else :
        cv2.putText(infoFrame, f"Coefficient de salissure : {colorDelta:.1f}", (0,int(xOffset)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    cv2.imshow(WINDOW_INFO_NAME, infoFrame)

def calculate_container_avg(measures):
    moyXpx = sum([m[0][0] for m in measures]) / len(measures)
    moyYpx = sum([m[0][1] for m in measures]) / len(measures)
    moyXcm = sum([m[1][0] for m in measures]) / len(measures)
    moyYcm = sum([m[1][1] for m in measures]) / len(measures)
    moyB   = sum([m[2][0] for m in measures]) / len(measures)
    moyG   = sum([m[2][1] for m in measures]) / len(measures)
    moyR   = sum([m[2][2] for m in measures]) / len(measures)
    return (moyXpx, moyYpx), (moyXcm, moyYcm), (moyB, moyG, moyR)
    
def camera_calibrate_disp(pos):
    posX = 0
    posY = 0
    circleRadiusPx = 20
    if pos == 0:
        #Reset values
        camera_calibrate_disp.image = np.zeros((300,300,3), dtype="uint8")
        camera_calibrate_disp.oldPosX = 0
        camera_calibrate_disp.oldPosY = 0
    #Turn old circle to green if it exist
    if(camera_calibrate_disp.oldPosX != 0) and (camera_calibrate_disp.oldPosY !=0) :
        cv2.circle(camera_calibrate_disp.image,
                   (int(camera_calibrate_disp.oldPosX), int(camera_calibrate_disp.oldPosY)),
                   circleRadiusPx, COLOR_GREEN, -1)
    match pos:
        case 0: # Bottom right
            posX = camera_calibrate_disp.image.shape[1] - circleRadiusPx
            posY = camera_calibrate_disp.image.shape[0] - circleRadiusPx
        case 1: # Top right
            posX = camera_calibrate_disp.image.shape[1] - circleRadiusPx
            posY = circleRadiusPx
        case 2: # Top left
            posX = circleRadiusPx
            posY = circleRadiusPx
        case 3: # Bottom left
            posX = circleRadiusPx
            posY = camera_calibrate_disp.image.shape[0] - circleRadiusPx
        case 4: # Center
            posX = camera_calibrate_disp.image.shape[1]/2
            posY = camera_calibrate_disp.image.shape[0]/2
        
        case _:
            print(f"Invalid position : {pos}")
            return 
    
    cv2.circle(camera_calibrate_disp.image, (int(posX), int(posY)), circleRadiusPx, COLOR_YELLOW, -1)
    cv2.imshow('Calibration POS', camera_calibrate_disp.image)
    camera_calibrate_disp.oldPosX = posX
    camera_calibrate_disp.oldPosY = posY

def camera_calibrate_compute(id, measurments):
    # Exit if no container detected
    if(id == -1): return
    
    camera_calibrate_compute.measure.append(measurments)
    lenArrMeas = len(camera_calibrate_compute.measure)
    print(f"Measure {lenArrMeas} added : Xpx,Ypx,Xcm,Ycm,B,G,R = {measurments}")
    
    if lenArrMeas >= 5:
        camera_calibrate_disp(0)
    else:
        camera_calibrate_disp(lenArrMeas)
        
    if lenArrMeas == 5 :
        # Get the average of all values
        dimPxMoy, dimCmMoy, colorMoy = calculate_container_avg(camera_calibrate_compute.measure)
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
        camera_calibrate_compute.measure = []
    
#################################
#           Main                #
#################################

# Script arguments
ap = argparse.ArgumentParser(description="Demo script for NUT project")
ap.add_argument("-c", "--calibration", action='store_true', required=False, help="Enter the calibration mode")
args = vars(ap.parse_args())
s_calibrationModeEn = args["calibration"]
if(s_calibrationModeEn == True):
    print("== Calibration mode enable. Dumping container is disabled ==")
else:
    print("== Calibration mode disable. Add '-c' or '--calibration' to enable ==")

##### INIT #####
print("===== Initialization NUT script =====")
### Screen ###
cv2.namedWindow(WINDOW_SCREEN_NAME)
cv2.moveWindow(WINDOW_SCREEN_NAME, 0, WINDOW_TASK_BAR_Y_OFFSET) # (x, y) #Move windows doesn't work on the raspbian, TODO : see why
# Get welcome page screen
screenImg = cv2.imread(SCREEN_NAME_WELCOME)
windowXOffset = screenImg.shape[1] + WINDOW_BORDER_X_OFFSET# Get X size to offset other windows
windowYOffset = screenImg.shape[0] # Get Y size to offset other windows
print(f"windowXOffset = {windowXOffset} | windowYOffset = {windowYOffset}")

### Windows ###
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
dispWidth = int(WINDOW_TERMINAL_X_OFFSET - windowXOffset)
dispHeight = int(dispWidth / cropRatio)
print(f"dispWidth = {dispWidth} | dispHeight = {dispHeight}")

### Waiting list ###
s_waitingOrderArr = waiting_list_read_file(FILE_OBJECT_ORDER_NAME)
print("Containers to detect :", s_waitingOrderArr)
s_waitingIndex = 0
# Init static variable in functions
waiting_process_id.updateDisp = True
waiting_process_id.detectionTime = 0
# waiting_display_id.text = str("")
# waiting_display_id.textX = 0
# waiting_display_id.textY = 0
# waiting_display_id.image = np.zeros((300,300,3), dtype="uint8") #3 is for the RGB (1 equal gray scale)

### GPIO ###
# Initialize limit switches as Button objects
s_switchStandby = Button(SWITCH_STANDBY_PIN, pull_up=True, bounce_time=0.01) #Set a small bounce time to still detect small press
s_switchOpen = Button(SWITCH_OPEN_PIN, pull_up=True, bounce_time=0.01)
# Attach callbacks to limit switches
s_switchStandby.when_pressed = gpio_switch_standby_cb
s_switchOpen.when_pressed = gpio_switch_open_cb

### Servo ###
# Initialize the ServoKit for 16 channels (Adafruit HAT)
#kit = ServoKit(channels=16)
#kit.servo[0].set_pulse_width_range(500, 2500) #From FS5109M datasheet
#kit.servo[0].actuation_range = 200 #From FS5109M datasheet
# Go to and find standby angle
#s_servoStandbyAngle = servo_init_position()
# print("Standby angle is : {:.2f}°".format(s_servoStandbyAngle))

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
camera_calculate_fps.fps = 0
camera_calculate_fps.framNb = 0
camera_calculate_fps.frameTime = time.time()
camera_calibrate_compute.measure = []
camera_calibrate_disp.image = np.zeros((300,300,3), dtype="uint8") #3 is for the RGB (1 equal gray scale)
camera_calibrate_disp.oldPosX = 0
camera_calibrate_disp.oldPosY = 0
print("===== Init done =====")

if(s_calibrationModeEn == True):
    camera_calibrate_disp(0)
    # update display
    cv2.waitKey(1)
else:
    # Display welcome page
    cv2.imshow(WINDOW_SCREEN_NAME, screenImg)
    # Wait any input
    print("Press any key to continue")
#     cv2.waitKey(0)
    cv2.waitKey(1)

# Create the camera and information windows
#Information is before to be behind the camera window
cv2.namedWindow(WINDOW_INFO_NAME)
cv2.moveWindow(WINDOW_INFO_NAME, windowXOffset, dispHeight+WINDOW_TASK_BAR_Y_OFFSET+WINDOW_OPENCV_BOT_Y_OFFSET) # (x, y)
infoFrame = np.ones((1080-(WINDOW_TASK_BAR_Y_OFFSET+dispHeight)-(WINDOW_OPENCV_TOP_Y_OFFSET+2*WINDOW_OPENCV_BOT_Y_OFFSET), dispWidth, 3), dtype=np.uint8) * 255
cv2.imshow(WINDOW_INFO_NAME, infoFrame)
#Camera
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, windowXOffset, WINDOW_TASK_BAR_Y_OFFSET) # (x, y)

##### LOOP #####
key = None
try:
    while(1):
        # Capture the image
        bgrFrame = camera.capture_array("main")
        # Analyze the image
        resizedFrame = cv2.resize(bgrFrame, PROCESSING_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        cropedFrame = resizedFrame[CROP_HEIGHT_PX_TOP:cropHeight, CROP_WIDTH_PX_LEFT:cropWidth]
        analyzedFrame, containerId, containerInfo, containerError = camera_find_contours(cropedFrame)
#         print("ContainerId = {}".format(containerId))
        # Resize image to take the maximum screen size
        displayedFrame = cv2.resize(analyzedFrame, (dispWidth, dispHeight),interpolation=cv2.INTER_LINEAR)
        # Display image
        camera_add_text_and_display(displayedFrame)
        # Get user inout key
        key = cv2.waitKey(1) & 0xFF
        # Quit if "q" orc ESC key is pressed
        if (key == ord('q')) or (key == 27):
            break
        
        if(s_calibrationModeEn == False):
            # Check if the correct container is detected and dump it
            waiting_process_id(containerId, containerError)
        else:
            if key == ord('c'):
                camera_calibrate_compute(containerId, containerInfo)  
                
except KeyboardInterrupt:
    print("===== Program interrupted =====")

finally:
    print("Cleaning the peripherals...")
    camera.stop()  # Stop the camera
    camera.close()  # Free resources
    cv2.destroyAllWindows()  # Close all OpenCV windows
    # gpio cleanup is automaticly done by gpiozero when script end
    print("===== Program ended ======")
