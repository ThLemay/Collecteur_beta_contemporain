print("Init fork.py...")

from gpiozero import Button
from time import sleep

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Initialisation I2C et PCA9685
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x40)
pca.frequency = 100

PIN_SWITCH_FORK_CENTER	= 23
PIN_SWITCH_FORK_LEFT	= 24

PWM_NB_MOTOR_IN1		= 2
PWM_NB_MOTOR_IN2		= 3

FORK_SPEED_CENTER	= 0.18 # Beteween 0 and 1 (0 = stop & 1 = max speed)
FORK_SPEED_DUMP_LEFT	= 0.18
FORK_SPEED_DUMP_RIGHT	= 0.18 # Not used
FORK_TIMEOUT_S			= 3

# End switches
switchForkCenter	= Button(PIN_SWITCH_FORK_CENTER,	bounce_time=0.01)	# Hardware pullup
switchForkDumpLeft	= Button(PIN_SWITCH_FORK_LEFT,		bounce_time=0.01)	# Hardware pullup

s_switchCenterTriggered 	= False
s_switchDumpLeftTriggered 	= False

ERROR_NONE		=  0
ERROR_TIMEOUT	= -1
ERROR_OBSTACLE	= -2

# Private functions ( "_" as a prefix)
def _switch_center_cb():
	stop()
	global s_switchCenterTriggered 
	s_switchCenterTriggered = True
	print("Fork center switch triggered")
	
def _switch_dump_left_cb():
	stop()
	global s_switchDumpLeftTriggered
	s_switchDumpLeftTriggered = True
	print("Fork dump left switch triggered")



def _forward(speed):
	if not 0 <= speed <= 1:
		raise ValueError('forward speed must be between 0 and 1')
	dutyCycle = int(0xFFFF * abs(speed))
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = dutyCycle
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = 0
	#print("Duty cycle forward is {:#x}".format(dutyCycle))

def _backward(speed):
	if not 0 <= speed <= 1:
		raise ValueError('backward speed must be between 0 and 1')
	dutyCycle = int(0xFFFF * abs(speed))
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = 0
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = dutyCycle
	#print("Duty cycle backward is {:#x}".format(dutyCycle))

# Public functions

def stop():
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = 0
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = 0
	print("Fork stopped")
	
def hold():
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = 0xFFFF
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = 0xFFFF
	print("Fork break")

def center():
	global s_switchCenterTriggered
	if(switchForkCenter.is_pressed):
		print("Fork already centered, exit")
		return ERROR_NONE
	print("Fork going to center position at {}% speed".format(FORK_SPEED_CENTER*100))
	s_switchCenterTriggered = False
	_backward(FORK_SPEED_CENTER)
	timeout = 0
	while((s_switchCenterTriggered == False) and (timeout < FORK_TIMEOUT_S)):
		sleep(0.01)
		timeout += 0.01
		#print("timeout={}, s_switchOpenTriggered={}".format(timeout, s_switchOpenTriggered))
	stop()
	#hold()
	if(timeout >= FORK_TIMEOUT_S):
		print("Fork timeout occured when going to center")
		return ERROR_TIMEOUT
	else:
		return ERROR_NONE
	
def dumpLeft():
	global s_switchDumpLeftTriggered
	if(switchForkDumpLeft.is_pressed):
		print("Fork already dumping on the left, exit")
		return ERROR_NONE
	print("Fork going to dumping on the left position at {}% speed".format(FORK_SPEED_DUMP_LEFT*100))
	s_switchDumpLeftTriggered = False
	_forward(FORK_SPEED_DUMP_LEFT)
	timeout = 0
	while((s_switchDumpLeftTriggered == False) and (timeout < FORK_TIMEOUT_S)):
		sleep(0.01)
		timeout += 0.01
		#print("timeout={}, s_switchCloseTriggered={}".format(timeout, s_switchCloseTriggered))
	stop()
	#hold()
	if(timeout >= FORK_TIMEOUT_S):
		print("Fork timeout occured when going to dumping on the left")
		return ERROR_TIMEOUT
	else:
		return ERROR_NONE
		
def dumpLeftAndReturnCenter():
    print("➡️ Fourche vers la gauche")
    err = dumpLeft()
    if err != ERROR_NONE:
        print("❌ Erreur sur dumpLeft()")
        return err
    print("⏳ Pause 0.5s")
    sleep(0.5)
    print("⬅️ Retour au centre (tempo 1.32s)")
    _backward(FORK_SPEED_CENTER)
    sleep(1.25)
    stop()
    return ERROR_NONE

def dumpRightAndReturnCenter():
    print("➡️ Fourche vers la droite")
    _backward(FORK_SPEED_DUMP_RIGHT)
    timeout = 0
    while not switchForkCenter.is_pressed and timeout < FORK_TIMEOUT_S:
        sleep(0.01)
        timeout += 0.01
    stop()
    if timeout >= FORK_TIMEOUT_S:
        print("⚠️ Capteur 'centre' non déclenché en allant à droite")
        return ERROR_TIMEOUT

    print("✅ Position droite atteinte (via capteur centre)")
    print("⏳ Pause 0.5s")
    sleep(0.5)
    print("⬅️ Retour au centre (tempo 1.35s)")
    _forward(FORK_SPEED_CENTER)
    sleep(1.40)
    stop()
    return ERROR_NONE

# Init callbacks
switchForkCenter.when_pressed 	= _switch_center_cb
switchForkDumpLeft.when_pressed = _switch_dump_left_cb

print("Init fork.py done")

if __name__ == "__main__":
	try:
		print("Test fork.py")
		while True:
			#continue;
			if(dumpRightAndReturnCenter()):
				print("Error while going to the center")
			print("Pause 5s")
			sleep(5)
			
			if(dumpLeftAndReturnCenter()):
				print("Error while going to dump on the left")	
			print("Pause 5s")
			sleep(5)
	
	except KeyboardInterrupt:
			print("Exiting program")
	
	finally:
		stop()
		pca.deinit()
		print("Cleanup complete")
