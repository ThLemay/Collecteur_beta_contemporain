print("Init trapdoor.py...")

from gpiozero import Button
from time import sleep

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Initialisation I2C et PCA9685
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x40)
pca.frequency = 100

PIN_STOP_TRAP_OPEN 	= 17
PIN_STOP_TRAP_CLOSE = 18
PIN_OPTIC_BARRIER	= 16

PWM_NB_MOTOR_IN1	= 0
PWM_NB_MOTOR_IN2	= 1

# Motor is 50 RPM
TRAP_SPEED_OPEN 	= 0.4 # Beteween 0 and 1 (0 = stop & 1 = max speed)
TRAP_SPEED_CLOSE	= 0.2
TRAP_TIMEOUT_S		= 2
TRAP_STAY_OPEN_S	= 3 # After the optic sensor is triggered while smart_close

# End switches
stopTrapOpen 	= Button(PIN_STOP_TRAP_OPEN, 	bounce_time=0.001)	# Hardware pullup
stopTrapClose 	= Button(PIN_STOP_TRAP_CLOSE,	bounce_time=0.001)	# Hardware pullup
opticBarrier 	= Button(PIN_OPTIC_BARRIER,  	pull_up = False)#, bounce_time=0.001)	# Hardware pulldown, disable RPi pull-up and set the correct logic

s_switchOpenTriggered = False
s_switchCloseTriggered = False
s_switchOpticBarrierTriggered = False

ERROR_NONE		= 0
ERROR_TIMEOUT	= -1
ERROR_OBSTACLE	= -2

# Private functions ( "_" as a prefix)
def _switch_open_cb():
	stop()
	global s_switchOpenTriggered 
	s_switchOpenTriggered = True
	print("Trap open switch triggered")
	
def _switch_close_cb():
	stop()
	global s_switchCloseTriggered
	s_switchCloseTriggered = True
	print("Trap close switch triggered")
	
def _switch_optic_barrier_cb():
	global s_switchOpticBarrierTriggered
	s_switchOpticBarrierTriggered = True
	print("Optic barrier triggered")

def _forward(speed):
	if not 0 <= speed <= 1:
		raise ValueError('forward speed must be between 0 and 1')
	dutyCycle = int(0xFFFF * abs(speed))
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = dutyCycle
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = 0

def _backward(speed):
	if not 0 <= speed <= 1:
		raise ValueError('backward speed must be between 0 and 1')
	dutyCycle = int(0xFFFF * abs(speed))
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = 0
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = dutyCycle

# Public functions

def stop():
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = 0
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = 0
	print("Trap stopped")
	
def hold():
	pca.channels[PWM_NB_MOTOR_IN1].duty_cycle = 0xFFFF
	pca.channels[PWM_NB_MOTOR_IN2].duty_cycle = 0xFFFF
	print("Trap break")

def open():
	global s_switchOpenTriggered
	if(stopTrapOpen.is_pressed):
		print("Trap already opened, exit")
		return ERROR_NONE
	print("Trap opening at {}% speed".format(TRAP_SPEED_OPEN*100))
	s_switchOpenTriggered = False
	_backward(TRAP_SPEED_OPEN)
	timeout = 0
	while((s_switchOpenTriggered == False) and (timeout < TRAP_TIMEOUT_S)):
		sleep(0.01)
		timeout += 0.01
		#print("timeout={}, s_switchOpenTriggered={}".format(timeout, s_switchOpenTriggered))
	stop()
	hold()
	if(timeout >= TRAP_TIMEOUT_S):
		print("Trap timeout occured when opening")
		return ERROR_TIMEOUT
	else:
		return ERROR_NONE
	
def close():
	global s_switchCloseTriggered, s_switchOpticBarrierTriggered
	if(stopTrapClose.is_pressed):
		print("Trap already closed, exit")
		return ERROR_NONE
	if(opticBarrier.is_pressed):
		print("Osbstacle in the way, cannot close, exit")
		return ERROR_OBSTACLE
	print("Trap closing at {}% speed".format(TRAP_SPEED_CLOSE*100))
	s_switchCloseTriggered = False
	s_switchOpticBarrierTriggered = False
	_forward(TRAP_SPEED_CLOSE)
	timeout = 0
	while((s_switchCloseTriggered == False) and (timeout < TRAP_TIMEOUT_S) and (s_switchOpticBarrierTriggered == False)):
		sleep(0.01)
		timeout += 0.01
		#print("timeout={}, s_switchCloseTriggered={}".format(timeout, s_switchCloseTriggered))
	stop()
	hold()
	if(timeout >= TRAP_TIMEOUT_S):
		print("Trap timeout occured when closing")
		return ERROR_TIMEOUT
	elif(s_switchOpticBarrierTriggered == True):
		print("Force stop, object detected by optic barrier")
		return ERROR_OBSTACLE
	else:
		return ERROR_NONE

def smart_close():
	err = close()
	while(err == ERROR_OBSTACLE):
		# Cannot close because the optic barrier saw something
  		# open and then wait for object to move to close it again
		if(open() == ERROR_TIMEOUT):
			return ERROR_TIMEOUT
		hold()
		while(opticBarrier.is_pressed):
			sleep(0.1)
		# Wait a bit before closing
		sleep(TRAP_STAY_OPEN_S)
		err = close()
	return err # ERROR_NONE or ERROR_TIMEOUT

# Init callbacks
stopTrapOpen.when_pressed = _switch_open_cb
stopTrapClose.when_pressed = _switch_close_cb
opticBarrier.when_pressed = _switch_optic_barrier_cb

print("Init trapdoor.py done")

if __name__ == "__main__":
	try:
		print("Test trapdoor.py")
		while True:
			if(open()):
				print("Error while opening")
			print("Pause 5s")
			sleep(5)
			
			if(smart_close()):
				print("Error while closing")	
			print("Pause 5s")
			sleep(5)
	
	except KeyboardInterrupt:
			print("Exiting program")
	
	finally:
		stop()
		pca.deinit()
		print("Cleanup complete")
