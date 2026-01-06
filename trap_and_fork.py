from time import sleep
# Local lib
import trapdoor
import fork

try:
	while True:
		if(trapdoor.smart_close()):
			print("Error when closing the trapdoor")
		sleep(1)
		if(fork.dumpLeft()):
			print("Error while going to dump on the left")
		sleep(1)
		if(fork.center()):
			print("Error while going to the center")
		sleep(1)
		if(trapdoor.open()):
			print("Error when opening the trapdoor")
		print("Start again in 5 seconds")
		sleep(5)
		
except KeyboardInterrupt:
	print("Exiting program")

finally:
	trapdoor.stop()
	fork.stop()
	print("Cleanup complete")
