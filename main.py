#Packages

import cv2

# Object detection method initialization

object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 50)

path = "images/testvid.mp4"

cap = cv2.VideoCapture(path)
while True: # Loop for the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))

    # Only getting the area of interest
    height, width, _ = frame.shape

    # Masking and finding the contours

    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 750:
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0),2)

    # Show the video
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)

    if key == ord("q"):  # lets you quit the video by pressing "q"
        break

cap.release()
cv2.destroyAllWindows()