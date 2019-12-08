import argparse
import cv2
import numpy as np
import imutils

ball_img = cv2.imread("bootstrap_img.png")
# convert it into HSV
ball_hsv = cv2.cvtColor(ball_img, cv2.COLOR_BGR2HSV)
# strip out zeros
ball_hsv_no_zeros = (ball_hsv[~np.any(ball_hsv == 0, axis=2)])
orange_min = ball_hsv_no_zeros[np.argmin(ball_hsv_no_zeros[:, 0]), :]
orange_max = np.array([np.max(ball_hsv_no_zeros[:, 0]), 255, 255], dtype=np.uint8)

def locate_ball(img, deback_img):
    scale = img.shape[1] / 600
    
    # frame_threshed = cv2.inRange(hsv, orange_min, orange_max)
    # cv2.imwrite('output.jpg', frame_threshed)
 
    # resize the frame, blur it, and convert it to the HSV
    # color space
    img = imutils.resize(img, width=600)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
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
    signals = []
    centers = []
    print('num features:', len(cnts))
    if len(cnts) < 1:
        return None
    mask = np.zeros_like(deback_img)
    for c in cnts:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        scenter_real = (center[0] * scale, center[1] * scale)
        scenter = (int(scenter_real[0]), int(scenter_real[1]))
        cv2.circle(mask, (int(scenter[0]), int(scenter[1])), 5, (255,255,255), -1)
        patch = deback_img[scenter[1]-5:scenter[1]+6, scenter[0]-5:scenter[0]+6, :]
        signal = np.sum(patch, axis=(0,1)).astype(np.float32)
        signal /= 121.0
        #print('signal level at', center, ":", signal)
        signals.append(np.linalg.norm(signal))
        centers.append(scenter_real)
    cv2.imshow("deback", cv2.add(deback_img, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    center = centers[np.argmax(signals)]
    return center
    

if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--frame", type=str, default="cropped_frame1.png",
                    help="input frame")
    ap.add_argument("-i", "--img", type=str, default="bootstrap_img.png",
                    help="image of the ball")
    args = vars(ap.parse_args())
    
    # Read ball img
    ball_img = cv2.imread(args["img"])
    
    
    # Read the given frame
    frame = cv2.imread(args["frame"])

    # convert it into HSV
    ball_hsv = cv2.cvtColor(ball_img, cv2.COLOR_BGR2HSV)
    
    # strip out zeros
    ball_hsv_no_zeros = (ball_hsv[~np.any(ball_hsv == 0, axis=2)])
    
    orange_min = ball_hsv_no_zeros[np.argmin(ball_hsv_no_zeros[:, 0]), :]
    orange_max = np.array([np.max(ball_hsv_no_zeros[:, 0]), 255, 255], dtype=np.uint8)
    
    # frame_threshed = cv2.inRange(hsv, orange_min, orange_max)
    # cv2.imwrite('output.jpg', frame_threshed)
    
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
    print(len(cnts))
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        #c = max(cnts, key=cv2.contourArea)
        c = cnts[0]
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        # Read the given frame
        #img = cv2.imread(args["frame"])
        #ass = locate_ball(img)
        #print(ass, center)
        #assert ass == center

        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv2.circle(frame, (int(x), int(y)), int(radius),
                   (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
