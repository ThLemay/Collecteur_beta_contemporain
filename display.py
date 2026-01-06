print("Init display.py...")

import cv2
import time
from time import sleep
import numpy as np
from camera import containerErrorCode, containerInfo_t
import camera

#################################
#           Defines             #
#################################

### Windows
WINDOW_SCREEN_NAME           = "0117-NUT"
WINDOW_BORDER_X_OFFSET       = 0
WINDOW_OPENCV_TOP_Y_OFFSET   = 50
WINDOW_OPENCV_BOT_Y_OFFSET   = 20
WINDOW_TASK_BAR_Y_OFFSET     = 62
WINDOW_TERMINAL_X_OFFSET     = 1425- WINDOW_BORDER_X_OFFSET #1920x1080
WINDOW_INFO_NAME             = "Container informations"
### Screen ####
SCREEN_FOLDER_PATH      = "screens/"
# Images name to display
SCREEN_NAME_WELCOME     = SCREEN_FOLDER_PATH + "Welcome_page.jpg"
SCREEN_NAME_ACCEPTED    = SCREEN_FOLDER_PATH + "Container_accepted.jpg"
SCREEN_NAME_REJECTED    = SCREEN_FOLDER_PATH + "Container_rejected.jpg"
SCREEN_NAME_CLOSING     = SCREEN_FOLDER_PATH + "Warning_trapdoor.jpg"
SCREEN_NAME_XS          = SCREEN_FOLDER_PATH + "Container_XS.jpg"
SCREEN_NAME_S           = SCREEN_FOLDER_PATH + "Container_S.jpg"
SCREEN_NAME_M           = SCREEN_FOLDER_PATH + "Container_M.jpg"
SCREEN_NAME_L           = SCREEN_FOLDER_PATH + "Container_L.jpg"
SCREEN_NAME_XL          = SCREEN_FOLDER_PATH + "Container_XL.jpg"
# Make a dictionnary of the container id to their name
CONTAINER_ID_TO_STRING  = {-1:"Non detecte", 0:"XS", 1:"S", 2:"M", 3:"L", 4:"XL"}

### Camera
WINDOW_CAMERA_NAME      = "Contour detection"

### Colors
# Colors are in BGR format
COLOR_BLUE      = (255, 0  , 0  )
COLOR_GREEN     = (0  , 255, 0  )
COLOR_RED       = (0  , 0  , 255)
COLOR_MAGENTA   = (255, 0  , 255)
COLOR_YELLOW    = (0  , 255, 255)
COLOR_CYAN      = (255, 255, 0  )
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0  , 0  , 0  )

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
   
def update():
    cv2.waitKey(1)
     
def screen_accepted():
    screen = cv2.imread(SCREEN_NAME_ACCEPTED)
    cv2.imshow(WINDOW_SCREEN_NAME, screen)
    
def screen_rejected():
    screen = cv2.imread(SCREEN_NAME_REJECTED)
    cv2.imshow(WINDOW_SCREEN_NAME, screen)
    
def screen_warning_closing():
    screen = cv2.imread(SCREEN_NAME_CLOSING)
    cv2.imshow(WINDOW_SCREEN_NAME, screen)
    
def _calculate_fps():
    # Calculate fps
    if (time.time() - _calculate_fps.frameTime) >= 1 :
        # 1 second elapsed, count the frame that 
        _calculate_fps.frameTime = time.time()
        _calculate_fps.fps = _calculate_fps.framNb
        _calculate_fps.framNb = 0
        # print("Current fps : {:2}".format(_calculate_fps.fps))
    else :
        _calculate_fps.framNb += 1

def camera_output(cameraOutputFrame, calibrationModeEn = False):
    # Resize
    resizedFrame = cv2.resize(cameraOutputFrame, (dispWidth, dispHeight),interpolation=cv2.INTER_LINEAR)
    # Transform nb of time this function is called to compute the fps
    _calculate_fps()
    # Add text
    cv2.putText(resizedFrame, "fps : {}".format(_calculate_fps.fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)
    cv2.putText(resizedFrame, "Press 'q' to exit", (0, resizedFrame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2) # Y picture size minus text size
    if(calibrationModeEn) :
        cv2.putText(resizedFrame, "Press 'c' to take a measurments", (0, resizedFrame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)
    # Display
    cv2.imshow(WINDOW_CAMERA_NAME, resizedFrame)
    return resizedFrame

def camera_detection_info(containerInfo: containerInfo_t):
    global infoFrame
    infoFrame = np.ones(infoFrame.shape, dtype=np.uint8) * 255
    xOffset = 30
    fontScale = 0.9
    if(containerInfo.error == containerErrorCode.NOT_RECOGNIZED) :
        cv2.putText(infoFrame, f"Contenant : {CONTAINER_ID_TO_STRING.get(containerInfo.id)}", (0,xOffset), cv2.FONT_ITALIC, fontScale, COLOR_RED, 2)
    else :
        cv2.putText(infoFrame, f"Contenant : {CONTAINER_ID_TO_STRING.get(containerInfo.id)}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    xOffset+= int((25 + 2*5) * fontScale) # Text size + 5 top and 5 bottom
    radius = (containerInfo.dimCm[0] + containerInfo.dimCm[1]) / 2
    if(containerInfo.error == containerErrorCode.MISSMATCH_SIZE) :
        cv2.putText(infoFrame, f"Dimension (cm) : {radius:.2f}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_RED, 2)
    else :
        cv2.putText(infoFrame, f"Dimension (cm) : {radius:.2f}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    xOffset+= int((25 + 2*5) * fontScale) # Text size + 5 top and 5 bottom
    cv2.putText(infoFrame, f"Couleur (format BVR) : {containerInfo.color}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    xOffset+= int((25 + 2*5) * fontScale) # Text size + 5 top and 5 bottom
    if(containerInfo.error == containerErrorCode.MISSMATCH_COLOR) :
        cv2.putText(infoFrame, f"Coefficient de salissure : {containerInfo.colorDelta:.1f}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC, fontScale, COLOR_RED, 2)
    else :
        cv2.putText(infoFrame, f"Coefficient de salissure : {containerInfo.colorDelta:.1f}", (0,xOffset), cv2.FONT_HERSHEY_SIMPLEX, fontScale, COLOR_BLACK, 2)
    cv2.imshow(WINDOW_INFO_NAME, infoFrame)
    
def get_key_input():
    return cv2.waitKey(1) & 0xFF
    
def begin(calibrationModeEn = False):
    if(calibrationModeEn == True):
        camera.calibrate_disp(0)
        # update display
        cv2.waitKey(1)
    else:
        # Display welcome page
        cv2.imshow(WINDOW_SCREEN_NAME, screenImg)
        # Wait any input
#       print("Press any key to continue")
#       cv2.waitKey(0)
        cv2.waitKey(1)
        
    cv2.imshow(WINDOW_INFO_NAME, infoFrame)

### Init static variable in functions ###
_calculate_fps.fps = 0
_calculate_fps.framNb = 0
_calculate_fps.frameTime = time.time()

### Screens ###
cv2.namedWindow(WINDOW_SCREEN_NAME)
cv2.moveWindow(WINDOW_SCREEN_NAME, 0, WINDOW_TASK_BAR_Y_OFFSET) # (x, y)
# Get welcome page screen
screenImg = cv2.imread(SCREEN_NAME_WELCOME)
windowXOffset = screenImg.shape[1] + WINDOW_BORDER_X_OFFSET# Get X size to offset other windows
windowYOffset = screenImg.shape[0] # Get Y size to offset other windows
print(f"windowXOffset = {windowXOffset} | windowYOffset = {windowYOffset}")

### Windows ###
# Process the display size from the cropped camera image
cropRatio = camera.get_crop_ratio()
dispWidth = int(WINDOW_TERMINAL_X_OFFSET - windowXOffset)
dispHeight = int(dispWidth / cropRatio)
print(f"dispWidth = {dispWidth} | dispHeight = {dispHeight}")



# Create the camera and information windows
#Information is before to be behind the camera window
cv2.namedWindow(WINDOW_INFO_NAME)
cv2.moveWindow(WINDOW_INFO_NAME, windowXOffset, dispHeight+WINDOW_TASK_BAR_Y_OFFSET+WINDOW_OPENCV_BOT_Y_OFFSET) # (x, y)
infoFrame = np.ones((1080-(WINDOW_TASK_BAR_Y_OFFSET+dispHeight)-(WINDOW_OPENCV_TOP_Y_OFFSET+2*WINDOW_OPENCV_BOT_Y_OFFSET), dispWidth, 3), dtype=np.uint8) * 255
#Camera
cv2.namedWindow(WINDOW_CAMERA_NAME)
cv2.moveWindow(WINDOW_CAMERA_NAME, windowXOffset, WINDOW_TASK_BAR_Y_OFFSET) # (x, y)

print("Init display.py done")


