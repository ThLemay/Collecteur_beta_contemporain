#################################
#           Includes            #
#################################

from gpiozero import Button
from adafruit_servokit import ServoKit
from time import sleep

#################################
#     Defines and Variables     #
#################################

# Define GPIO pins for the limit switches
SWITCH_STANDBY_PIN  = 17 # Limit switch standby
SWITCH_OPEN_PIN     = 18 # Limit switch open

# Servo motor
SERVO_CHANNEL           = 0
DELAY_BEFORE_RETURN_SEC = 3

s_switchStandbyTriggered = False
s_switchOpenTriggered = False

s_servoStandbyAngle = 90; # Value where we are sure neither of the switches will be triggered

#################################
#           Functions           #
#################################

# Callback for the gpio
def switchStandby_cb():
    global s_switchStandbyTriggered
    s_switchStandbyTriggered = True
    print("Stanby switch triggered")
    
def switchOpen_cb():
    global s_switchOpenTriggered
    s_switchOpenTriggered = True
    print("Open switch triggered")

def dump_container():
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

#################################
#           Main                #
#################################

# Initialize limit switches as Button objects
s_switchStandby = Button(SWITCH_STANDBY_PIN, pull_up=True, bounce_time=0.01) #Set a small bounce time to still detect small press
s_switchOpen = Button(SWITCH_OPEN_PIN, pull_up=True, bounce_time=0.01)
# Attach callbacks to limit switches
s_switchStandby.when_pressed = switchStandby_cb
s_switchOpen.when_pressed = switchOpen_cb

# Initialize the ServoKit for 16 channels (Adafruit HAT)
kit = ServoKit(channels=16)
kit.servo[0].set_pulse_width_range(500, 2500) #From FS5109M datasheet
kit.servo[0].actuation_range = 200 #From FS5109M datasheet

# try to find standby angle
currentAngle = kit.servo[0].angle
if(currentAngle == None):
    # This case happen the first time the servo is used
    print("Warning : read angle is null !")
    currentAngle = 90
while(s_switchStandby.is_pressed == False):
    currentAngle -= 0.5
    kit.servo[0].angle = currentAngle
    sleep(0.05)
    
s_servoStandbyAngle = kit.servo[0].angle
print("Standby angle is : {:.2f}°".format(s_servoStandbyAngle))

print("Init done")

try:
    # Move the sero to dump the container
    dump_container()
#     while(1): sleep(1)

except KeyboardInterrupt:
    print("Program interrupted")

finally:
    print("Program ended")
    # gpio cleanup is automaticly done by gpiozero when script end