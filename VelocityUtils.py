import numpy as np
import cv2

# Default parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def with_LK_optical_flow(curr_frame, next_frame, track_points_list, delta_t, params=lk_params):

    track_points = np.array([track_points_list], dtype=np.float32)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    track_points2, st, err = cv2.calcOpticalFlowPyrLK(curr_frame_gray, next_frame_gray, track_points, None, **params)

    good_new = track_points2[st == 1]
    good_old = track_points[st == 1]
    if len(good_new) != 1:
        raise RuntimeError('LK optical flow failed to find velocity')
    old, new = good_old[0], good_new[0]

    # calculate velocity at ball_position (directly use optical flow for now),
    # then the trajectory. Note the unit of time here is frame, so unit of
    # velocity is pixel/frame.
    a, b = new.ravel()
    c, d = old.ravel()
    velocity = ((a - c) / delta_t, (b - d) / delta_t)
    return velocity
