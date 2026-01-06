import cv2
import time
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

image = cv2.imread("../screens/Welcome_page.jpg")

cv2.namedWindow('Test', cv2.WINDOW_AUTOSIZE)

cv2.moveWindow('Test', 0, 0)

# image = (255, 255, 255)
cv2.imshow('Test', image)

cv2.waitKey(1)

for i in range(50):
    cv2.moveWindow('Test', i*2, i*2)
    cv2.waitKey(1)
    time.sleep(0.05)
    
for i in range(50):
    cv2.moveWindow('Test', 100+i*2, 100-i*2)
    cv2.waitKey(1)
    time.sleep(0.05)

# cv2.waitKey(0)
cv2.destroyAllWindows()
