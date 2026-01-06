
import time
from time import sleep
import argparse 
import trapdoor
import fork
import display
import camera
from camera import containerErrorCode, containerInfo_t
import cv2
import sys

# Flag global pour démarrer la détection
detection_started = False

# Callback souris : passe le flag à True sur clic gauche
def on_mouse(event, x, y, flags, param):
    global detection_started
    if event == cv2.EVENT_LBUTTONDOWN:
        detection_started = True
        print("[INFO] Clic détecté → détection activée")

s_calibrationModeEn = False
MAIN_ENABLE_MOTOR = True

FILE_OBJECT_ORDER_NAME = "awaiting_object_list.txt"
WAITING_VALIDATION_DELAY_S = 0.3

def dump_container(container_id: int):
    if trapdoor.smart_close():
        print("❌ Erreur lors de la fermeture intelligente de la trappe")
        return

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

    sleep(1)

    if trapdoor.open():
        print("❌ Erreur lors de l’ouverture de la trappe après le dump")

def waiting_list_read_file(filePath):
    with open(filePath, 'r') as f:
        return [int(l.strip()) for l in f.readlines()]

def waiting_process_id(containerInfo: containerInfo_t):
    global detection_started

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
                    dump_container(containerInfo.id)
                else:
                    print("Simulation des mouvements")
                    display.screen_accepted()
                    display.update()
                    sleep(2)

                detection_started = False
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

# === Script arguments ===
ap = argparse.ArgumentParser(description="Demo script for NUT project")
ap.add_argument("-c", "--calibration", action='store_true', help="Enter calibration mode")
args = vars(ap.parse_args())
s_calibrationModeEn = args["calibration"]

if s_calibrationModeEn:
    print("== Calibration mode enabled. Dumping container is disabled ==")
else:
    print("== Calibration mode disabled. Use '-c' to enable calibration mode ==")

# === Init ===
camera.s_waitingOrderArr = waiting_list_read_file(FILE_OBJECT_ORDER_NAME)
print("Containers to detect:", camera.s_waitingOrderArr)
camera.s_waitingIndex = 0
waiting_process_id.updateDisp = True
waiting_process_id.detectionTime = 0

display.begin(s_calibrationModeEn)
cv2.setMouseCallback("0117-NUT", on_mouse)

if MAIN_ENABLE_MOTOR:
    print("[INIT] Initialisation de la position des actionneurs…")
    if fork.center() != fork.ERROR_NONE:
        print("⚠️ Erreur de centrage, tentative de récupération manuelle")
        fork._forward(0.18)
        sleep(1.5)
        fork.stop()
        fork.center()
    if trapdoor.open() != trapdoor.ERROR_NONE:
        print("⚠️ Trappe n’a pas pu s’ouvrir correctement")

# Affichage initial pour dimensionner correctement les fenêtres
bgrFrame = camera.get_frame()
resizedFrame = camera.resize_and_crop_frame(bgrFrame)
analyzedFrame, containerInfo = camera.find_contours(resizedFrame, s_calibrationModeEn)
display.camera_detection_info(containerInfo)
display.camera_output(analyzedFrame, s_calibrationModeEn)

print("⌛ En attente du clic pour démarrer la détection…")
while not detection_started:
    frame = camera.get_frame()
    resized = camera.resize_and_crop_frame(frame)
    cv2.imshow("Contour detection", resized)
    if MAIN_ENABLE_MOTOR:
        trapdoor.close()
    if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
        print("Sortie avant démarrage.")
        sys.exit(0)

# Détection active
if MAIN_ENABLE_MOTOR:
    trapdoor.open()
print("🔓 Trappe ouverte, détection active !")

try:
    while True:
        if not detection_started:
            print("🟡 Pause. Cliquez pour continuer...")
            frame = camera.get_frame()
            resized = camera.resize_and_crop_frame(frame)
            cv2.imshow("0117-NUT", resized)
            if MAIN_ENABLE_MOTOR:
                trapdoor.close()
            if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
                break
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
            if not detection_started:
                continue
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
