import time
from time import sleep
# For script arguments
import argparse 
# Local lib
import trapdoor
import fork
import display
import camera
from camera import containerErrorCode, containerInfo_t

import cv2

# Flag global pour démarrer la détection
detection_started = False

# Callback souris : passe le flag à True sur clic gauche
def on_mouse(event, x, y, flags, param):
    global detection_started
    if event == cv2.EVENT_LBUTTONDOWN:
        detection_started = True

s_calibrationModeEn = False

MAIN_ENABLE_MOTOR = True

### Waiting list ###
FILE_OBJECT_ORDER_NAME = "awaiting_object_list.txt"
WAITING_VALIDATION_DELAY_S = 0.3 # container need to be seen during this period to be validated and dump

def dump_container(container_id: int):
    if trapdoor.smart_close():
        print("❌ Erreur lors de la fermeture intelligente de la trappe")
        return

    # Affichage écran de succès
    display.screen_accepted()
    display.update()

    if container_id in [0, 1]:
        print(f"➡️ Contenant ID {container_id} → tri à droite")
        err = fork.dumpRightAndReturnCenter()
    elif container_id in [2, 3]:
        print(f"⬅️ Contenant ID {container_id} → tri à gauche")
        err = fork.dumpLeftAndReturnCenter()
    else:
        print(f"❓ ID {container_id} inconnu → aucun mouvement effectué")
        sleep(1)
        err = fork.ERROR_NONE

    if err != fork.ERROR_NONE:
        print("❌ Erreur pendant le mouvement de la fourche")

    sleep(1)  # Délai après le dump

    if trapdoor.open():
        print("❌ Erreur lors de l’ouverture de la trappe après le dump")

def wait_for_user_click():
    global detection_started
    detection_started = False
    print("🟡 Pause. Cliquez pour continuer...")

    while not detection_started:
        frame = camera.get_frame()
        resized = camera.resize_and_crop_frame(frame)
        cv2.imshow("Contour detection", resized)
        if MAIN_ENABLE_MOTOR:
            trapdoor.close()
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            print("Sortie utilisateur pendant la pause")
            sys.exit(0)

"""
def dump_container():
    if(trapdoor.smart_close()):
        print("Error when closing the trapdoor")
    # Trap is now closed, we can display the point gained
    display.screen_accepted()
    display.update()
    # Continue the dump process
    if(fork.dumpLeft()):
        print("Error while going to dump on the left")
    sleep(1)
    if(fork.center()):
        print("Error while going to the center")
    if(trapdoor.open()):
        print("Error when opening the trapdoor")"""

def waiting_list_read_file(filePath):
    objectsOrder = []
    with open(filePath, 'r') as f:
       objectsOrder = [int(ligne.strip()) for ligne in f.readlines()]
    return objectsOrder 

def waiting_process_id(containerInfo: containerInfo_t):
    if(containerInfo.id != -1): # A known object is found
        #Go back to yellow if no object are found
        waiting_process_id.updateDisp = True
        if((containerInfo.id == camera.s_waitingOrderArr[camera.s_waitingIndex]) and (containerInfo.error == containerErrorCode.VALID)):
            # First time 
            if(waiting_process_id.detectionTime == 0):
                waiting_process_id.detectionTime = time.time()
                print("     Correct input container n°{}".format(containerInfo.id))
                display.screen_waiting_container_disp(camera.s_waitingOrderArr[camera.s_waitingIndex])
                # waiting_display_id(s_waitingOrderArr[s_waitingIndex], COLOR_GREEN)
                # update display
                # cv2.waitKey(1)
            # Container need to be seen WAITING_VALIDATION_DELAY_S seconds to be validated and dump
            if (time.time() - waiting_process_id.detectionTime) >= WAITING_VALIDATION_DELAY_S :
                # update display
                display.screen_warning_closing()
                display.update()
                sleep(1)
                # Dump container
                if(MAIN_ENABLE_MOTOR):
                    dump_container(containerInfo.id)
                else:
                    print("Simulate dumping")
                    sleep(1.5)
                    display.screen_accepted()
                    display.update()
                    sleep(2)
		    
                # Repasser en pause : attendre un clic utilisateur
                waiting_process_id.detection_started = False
                print("[INFO] Conteneur traité. Cliquez pour continuer.")
		
		# Check next container to get
                camera.s_waitingIndex+=1
                if camera.s_waitingIndex == len(camera.s_waitingOrderArr) : 
                    camera.s_waitingIndex = 0
                    print("     Loop waiting list!")
        else:
            # waiting_display_id(s_waitingOrderArr[s_waitingIndex], COLOR_RED)
            display.screen_rejected()
            print("Wrong input container ! n°{} instead of n°{}".format(containerInfo.id, camera.s_waitingOrderArr[camera.s_waitingIndex]))  
            # Reset the detection duration
            waiting_process_id.detectionTime = 0          
    else:
        # Do once, put current or new container ID in yellow
        if(waiting_process_id.updateDisp == True) :
            waiting_process_id.updateDisp = False
            # waiting_display_id(s_waitingOrderArr[s_waitingIndex], COLOR_YELLOW)
            display.screen_waiting_container_disp(camera.s_waitingOrderArr[camera.s_waitingIndex])
            # Reset the detection duration
            waiting_process_id.detectionTime = 0 

# Script arguments
ap = argparse.ArgumentParser(description="Demo script for NUT project")
ap.add_argument("-c", "--calibration", action='store_true', required=False, help="Enter the calibration mode")
args = vars(ap.parse_args())
s_calibrationModeEn = args["calibration"]
if(s_calibrationModeEn == True):
    print("== Calibration mode enable. Dumping container is disabled ==")
else:
    print("== Calibration mode disable. Add '-c' or '--calibration' to enable ==")

### Waiting list ###
camera.s_waitingOrderArr = waiting_list_read_file(FILE_OBJECT_ORDER_NAME)
print("Containers to detect :", camera.s_waitingOrderArr)
camera.s_waitingIndex = 0
# Init static variable in functions
waiting_process_id.updateDisp = True
waiting_process_id.detectionTime = 0

# Start displaying screens
display.begin(s_calibrationModeEn)

if(MAIN_ENABLE_MOTOR):
    print("Set default position for trap and for")
    fork.dumpRightAndReturnCenter()
    trapdoor.open()
   

cv2.setMouseCallback("0117-NUT", on_mouse)   

print("⌛ En attente du clic pour démarrer la détection…")
while not detection_started:
    # 1. Affichez le flux (mettez à jour votre fenêtre caméra)
# Force un premier affichage complet pour init les fenêtres aux bonnes dimensions
    bgrFrame = camera.get_frame()
    resizedFrame = camera.resize_and_crop_frame(bgrFrame)
    analyzedFrame, containerInfo = camera.find_contours(resizedFrame, s_calibrationModeEn)
    display.camera_detection_info(containerInfo)
    display.camera_output(analyzedFrame, s_calibrationModeEn)

    # 2. Verrouillez la trappe
    if MAIN_ENABLE_MOTOR:
        trapdoor.close()
    # 3. Quitter si nécessaire
    if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
        print("Sortie avant démarrage.")
        sys.exit(0)
# À la sortie de cette boucle, detection_started == True
if MAIN_ENABLE_MOTOR:
    trapdoor.open()
print("🔓 Trappe ouverte, détection active !")

try:
	while True:
		# Does the same as the 3 following lines : "analyzedFrame, containerInfo = camera.get_and_analyze_frame()"
		# Compute camera output
		bgrFrame = camera.get_frame()
		resizedFrame = camera.resize_and_crop_frame(bgrFrame)
		analyzedFrame, containerInfo = camera.find_contours(resizedFrame, s_calibrationModeEn)
		# Display detection informations
		display.camera_detection_info(containerInfo)
		display.camera_output(analyzedFrame, s_calibrationModeEn)
		# Check user input
		key = display.get_key_input()
		# Quit if "q" orc ESC key is pressed
		if (key == ord('q')) or (key == 27):
			break
		# Compute container
		if(s_calibrationModeEn == False):
			# Check if the correct container is detected and dump it
			waiting_process_id(containerInfo)
		else:
			if key == ord('c'):
				containerDim = containerInfo.dimPx, containerInfo.dimCm, containerInfo.color
				camera.calibrate_compute(containerInfo.id, containerDim)  
		
except KeyboardInterrupt:
	   print("Exiting program")
	   
finally:
    trapdoor.stop()
    camera.uninit()
    print("Cleanup complete")
   
"""
import time
from time import sleep
import argparse 
import trapdoor
import fork
import display
import camera
from camera import containerErrorCode, containerInfo_t
import cv2

# Variable de contrôle du clic
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("[INFO] Clic détecté → démarrage de la détection")
        waiting_process_id.detection_started = True

# Calibration
s_calibrationModeEn = False
MAIN_ENABLE_MOTOR = True

# Fichier des objets attendus
FILE_OBJECT_ORDER_NAME = "awaiting_object_list.txt"
WAITING_VALIDATION_DELAY_S = 1

def dump_container():
    if(trapdoor.smart_close()):
        print("Error when closing the trapdoor")
    display.screen_accepted()
    display.update()
    if(fork.dumpLeft()):
        print("Error while going to dump on the left")
    sleep(1)
    if(fork.center()):
        print("Error while going to the center")
    if(trapdoor.open()):
        print("Error when opening the trapdoor")

def waiting_list_read_file(filePath):
    objectsOrder = []
    with open(filePath, 'r') as f:
        objectsOrder = [int(ligne.strip()) for ligne in f.readlines()]
    return objectsOrder 

def waiting_process_id(containerInfo: containerInfo_t):
    if containerInfo.id != -1:
        waiting_process_id.updateDisp = True
        if (containerInfo.id == camera.s_waitingOrderArr[camera.s_waitingIndex]) and (containerInfo.error == containerErrorCode.VALID):
            if waiting_process_id.detectionTime == 0:
                waiting_process_id.detectionTime = time.time()
                print("     Correct input container n°{}".format(containerInfo.id))
                display.screen_waiting_container_disp(camera.s_waitingOrderArr[camera.s_waitingIndex])

            if (time.time() - waiting_process_id.detectionTime) >= WAITING_VALIDATION_DELAY_S:
                display.screen_warning_closing()
                display.update()
                sleep(1)

                if MAIN_ENABLE_MOTOR:
                    dump_container()
                else:
                    print("Simulate dumping")
                    sleep(1.5)
                    display.screen_accepted()
                    display.update()
		    sleep(2)

                # Repasser en pause : attendre un clic utilisateur
                waiting_process_id.detection_started = False
                print("[INFO] Conteneur traité. Cliquez pour continuer.")

                camera.s_waitingIndex += 1
                if camera.s_waitingIndex == len(camera.s_waitingOrderArr):
                    camera.s_waitingIndex = 0
                    print("     Loop waiting list!")
        else:
            display.screen_rejected()
            print("Wrong input container ! n°{} instead of n°{}".format(containerInfo.id, camera.s_waitingOrderArr[camera.s_waitingIndex]))
            waiting_process_id.detectionTime = 0
    else:
        if waiting_process_id.updateDisp:
            waiting_process_id.updateDisp = False
            display.screen_waiting_container_disp(camera.s_waitingOrderArr[camera.s_waitingIndex])
            waiting_process_id.detectionTime = 0

# Args
ap = argparse.ArgumentParser(description="Demo script for NUT project")
ap.add_argument("-c", "--calibration", action='store_true', required=False, help="Enter the calibration mode")
args = vars(ap.parse_args())
s_calibrationModeEn = args["calibration"]
if s_calibrationModeEn:
    print("== Calibration mode enabled. Dumping container is disabled ==")
else:
    print("== Calibration mode disabled. Add '-c' or '--calibration' to enable ==")

# Initialisation
camera.s_waitingOrderArr = waiting_list_read_file(FILE_OBJECT_ORDER_NAME)
print("Containers to detect :", camera.s_waitingOrderArr)
camera.s_waitingIndex = 0
waiting_process_id.updateDisp = True
waiting_process_id.detectionTime = 0
waiting_process_id.detection_started = False

display.begin(s_calibrationModeEn)
cv2.setMouseCallback("Display", on_mouse_click)
print("[INFO] Cliquez sur la fenêtre 'Display' pour démarrer la détection")

if MAIN_ENABLE_MOTOR:
    print("Set default position for trap and fork")
    fork.center()
    trapdoor.open()

try:
    while True:
        if not waiting_process_id.detection_started:
            sleep(0.1)
            continue

        bgrFrame = camera.get_frame()
        resizedFrame = camera.resize_and_crop_frame(bgrFrame)
        analyzedFrame, containerInfo = camera.find_contours(resizedFrame, s_calibrationModeEn)

        display.camera_detection_info(containerInfo)
        display.camera_output(analyzedFrame, s_calibrationModeEn)

        key = display.get_key_input()
        if key == ord('q') or key == 27:
            break

        if not s_calibrationModeEn:
            waiting_process_id(containerInfo)
        else:
            if key == ord('c'):
                containerDim = containerInfo.dimPx, containerInfo.dimCm, containerInfo.color
                camera.calibrate_compute(containerInfo.id, containerDim)

except KeyboardInterrupt:
    print("Exiting program")

finally:
    trapdoor.stop()
    camera.uninit()
    print("Cleanup complete") """


