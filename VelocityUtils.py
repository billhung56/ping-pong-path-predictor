import numpy as np
import cv2

# Default parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def with_LK_optical_flow(curr_frame, next_frame, track_points_list, params=lk_params):

    track_points = np.array(track_points_list, dtype=np.float32)
    track_points = track_points.reshape(-1,1,2)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    track_points2, st, err = cv2.calcOpticalFlowPyrLK(curr_frame_gray, next_frame_gray, track_points, None, **params)

    good_new = track_points2[st == 1]
    good_old = track_points[st == 1]
    if len(good_new) < 1:
        raise RuntimeError('LK optical flow failed to find velocity')
    vs = []
    for (new,old) in zip(good_new, good_old):
        # calculate velocity at ball_position (directly use optical flow for now),
        # then the trajectory. Note the unit of time here is frame, so unit of
        # velocity is pixel/frame.
        a, b = new.ravel()
        c, d = old.ravel()
        v = (a - c, b - d)
        vs.append(v)
    return np.mean(vs, axis=0)

def with_LK_optical_flow_N(frames, track_points_list, params=lk_params):
    assert len(frames) >= 2
    track_points = np.array(track_points_list, dtype=np.float32)
    track_points = track_points.reshape(-1,1,2)
    curr_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    frame_vs = []
    for i in range(1, len(frames)):
        next_frame = frames[i]
        next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        track_points2, st, err = cv2.calcOpticalFlowPyrLK(curr_frame_gray, next_frame_gray, track_points, None, **params)

        good_new = track_points2[st == 1]
        good_old = track_points[st == 1]
        if len(good_new) < 1:
            raise RuntimeError('LK optical flow failed to find velocity')
        vs = []
        for (new,old) in zip(good_new, good_old):
            # calculate velocity at ball_position (directly use optical flow for now),
            # then the trajectory. Note the unit of time here is frame, so unit of
            # velocity is pixel/frame.
            a, b = new.ravel()
            c, d = old.ravel()
            v = (a - c, b - d)
            vs.append(v)
        vel = np.mean(vs, axis=0)
        frame_vs.append(vel)

        curr_frame_gray = next_frame_gray.copy()
        track_points = good_new.reshape(-1,1,2)
    return frame_vs

fb_params = [
    0.5,     # pyr_scale
    3,       # levels
    15,      # winsize
    3,       # iterations
    7,       # poly_n
    1.5,     # poly_sigma
    0        # flags
]

def with_FB_optical_flow(curr_frame, next_frame, track_points_list, delta_t, params=fb_params):
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(curr_frame_gray, next_frame_gray, None, *fb_params)
    p = (int(track_points_list[0][0]), int(track_points_list[0][1]))
    flow_patch = flow[p[1]-3:p[1]+3, p[0]-3:p[0]+3]
    velocity = np.sum(flow_patch, axis=(0,1)) / 49.0
    return velocity
