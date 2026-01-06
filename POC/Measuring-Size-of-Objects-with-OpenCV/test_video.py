from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder 
from picamera2.outputs import FfmpegOutput
import cv2
import time

# Configure one camera (model 3)
camera = Picamera2(camera_num=0)

# Create the configuration
camera_config = camera.create_video_configuration(
    main={"size": (1280, 720)},  # Base resolution in 1080p (can be adjusted, max is : 4608,2592)
    lores={"size": (640, 480)},   # Low resolution for previsualisation
)
camera.configure(camera_config)
encoder = H264Encoder(10000000)

def record_and_display_video(duration):
    try:      
        # Create output
        video_output = FfmpegOutput("Test_Video.mp4")
        
        # Start recording
        camera.start_recording(encoder,output=video_output)  # Start video recording
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Save current image (main feed, not the preview one to display the actual quality)
            frame_bgr = camera.capture_array("main")
            # Image saved is in BGR format, covert it to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Display image with OpenCV
            cv2.imshow("Recording Video", frame_rgb)
            # Quit if "q" is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop recording
        camera.stop_recording()
    finally:
        camera.stop()  # Stop the camera
        camera.close()  # Free resources
        cv2.destroyAllWindows()  # Close all OpenCV windows

# Film and display a video for 10s
record_and_display_video(10)
print("Video captured, displayed in real time and ressources are freed")
