import argparse
import cv2
import numpy as np
import imutils

# np.set_printoptions(threshold=np.inf)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--frame", type=str, default="cropped_frame1.png",
                help="input frame")
ap.add_argument("-i", "--img", type=str, default="bootstrap_img.png",
                help="image of the ball")
args = vars(ap.parse_args())

# Read ball img
ball_img = cv2.imread(args["img"])

# convert it into HSV
ball_hsv = cv2.cvtColor(ball_img, cv2.COLOR_BGR2HSV)

# strip out zeros
ball_hsv_no_zeros = (ball_hsv[~np.any(ball_hsv == 0, axis=2)])

orange_min = ball_hsv_no_zeros[np.argmin(ball_hsv_no_zeros[:, 0]), :]
orange_max = np.array([np.max(ball_hsv_no_zeros[:, 0]), 255, 255], dtype=np.uint8)

# frame_threshed = cv2.inRange(hsv, orange_min, orange_max)
# cv2.imwrite('output.jpg', frame_threshed)

# Read the given frame
frame = cv2.imread(args["frame"])

# resize the frame, blur it, and convert it to the HSV
# color space
frame = imutils.resize(frame, width=600)
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# construct a mask for the color "orange", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, orange_min, orange_max)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# find contours in the mask and initialize the current
# (x, y) center of the ball
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
center = None

# only proceed if at least one contour was found
if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # draw the circle and centroid on the frame,
    # then update the list of tracked points
    cv2.circle(frame, (int(x), int(y)), int(radius),
               (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
