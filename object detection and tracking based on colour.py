import cv2
import imutils
import numpy as np

# HSV range (yellow/green range, not pure red)
lower1 = (0, 120, 70)
upper1 = (10, 255, 255)

lower2 = (170, 120, 70)
upper2 = (180, 255, 255)
camera = cv2.VideoCapture(1)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0 and radius > 10:
            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            )

            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # 🚗 Direction logic
            if radius > 120:
                print("STOP")
            elif center[0] < 150:
                print("RIGHT")
            elif center[0] > 450:
                print("LEFT")
            else:
                print("FRONT")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
