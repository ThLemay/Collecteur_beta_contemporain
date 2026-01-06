from picamera2 import Picamera2, Preview
import cv2
import time

# Init camera
camera = Picamera2(camera_num=0)

# Set config for camera
camera_config = camera.create_still_configuration(
    main={"size": (4608, 2592)},  # Max resolution
    lores={"size": (640, 480)},   # Optionnal : low quality feed with low resolution
    display="lores"               # Use low quality for preview
)
camera.configure(camera_config)

# Start the camera
camera.start()
time.sleep(2) # Let time to start

image_output_path = "high_res_image.jpg"

def preview_and_capture():
    try:
        while True:
            # Use the low resolution for preview
            frame = camera.capture_array("lores")
            # Image saved is in BGR format, covert it to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display with openCV
            cv2.imshow("Camera Preview", frame_rgb)

            # Check if 'c' is pressed to take a screenshot
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Screenshot and save image
                camera.capture_file(image_output_path)
                print("Picture saved at %s" % image_output_path)
                break

            # Quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capture annulée.")
                break
    finally:
        camera.stop()  # Stop camera
        camera.close()  # Free camera ressources
        cv2.destroyAllWindows()  # Close all OpenCV windows

# Start the preview
preview_and_capture()
# End
