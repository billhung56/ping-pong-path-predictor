""" Utility program for project ball-ping-pong

This program takes a video of a players playing ping pong (need
not include the players) and predicts the trajectory of the ball,
with only partial frames of the video. The basic idea is to run
optical flow over two adjacent (or close-in-time) frames to estimate
a velocity vector, and compute a curve (parabola for now) based
on the velocity as a function of time.

For usage of this program, run "python3 main.py -h"

args:
  video: Path to the video that has a ping pong ball moving.
  start: Start time to used for prediction, also adopted as the start of trajectory.
  duration: How long the trajectory is in terms of time.

optional args:
  delta: An integer such that the delta_t-th frame after the start frame is used
         as the second frame of optical flow. Default to 1.
"""
import argparse
from datetime import time
import numpy as np
import cv2
from PIL import Image
import time as ctime
import locate_ball

# gravity in pixel/sec. need tuning for different camera Orz.
base_ng = 0.0038  # (for rvid)
#base_ng = 0.00215  # or 0.0024 (for vid)
def compute_trajectory(p, v, ng):
    """compute poly^2 trajectory from position and velocity
       Returns x(t), y(t)
        p: position/pose (p_x, p_y)
        v: velocity (v_x, v_y) (pixel/frame)
        ng: gravity (pixel/frame^2)
    """
    xOt = np.polynomial.polynomial.Polynomial((p[0], v[0]))
    yOt = np.polynomial.polynomial.Polynomial((p[1], v[1], ng/2))
    return xOt, yOt

def extract_background(vidpath):
    cap = cv2.VideoCapture(vidpath)

    # Randomly select 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
     
    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)
     
    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
    return medianFrame
     

if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("video", help="path to video of ball movement")
    cmd_parser.add_argument("--start", help="time to start drawing the trajectory and used for prediction (in ISO format)", default="00:00:00")
    cmd_parser.add_argument("--duration", help="duration of the trajectory to predict (in ISO format)", default="00:00:01")
    cmd_parser.add_argument("--delta", type=int, choices=range(1,11), help="frame difference for optical flow", default=1)
    args = cmd_parser.parse_args()

    # Display background
    background = extract_background(args.video)
    cv2.imshow('background', background)

    cap = cv2.VideoCapture(args.video)
    # get info about video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'frame rate: {frame_rate}')
    print(f'total frames: {total_frames}')
    print(f'frame size: {int(vid_width)} x {int(vid_height)}')
    start, duration = None, None

    # addtional processing to validate arguments
    try:
        start = time.fromisoformat(args.start)
    except ValueError:
        raise ValueError("argument \"start\" not specified in appropriate format. maybe try ISO format?")
    # calculate corresponding frame id at time start
    start_sec = start.hour * 3600 + start.minute * 60 + start.second + start.microsecond / 1000000.0
    start_frame = int(np.ceil(start_sec / (1.0 / frame_rate)))
    # check start in video, else raise error
    if start_frame < 0 or start_frame >= total_frames - args.delta:
        raise ValueError("start time out of bound")
    start_frame += 1

    try:
        duration = time.fromisoformat(args.duration)
    except ValueError:
        raise ValueError("argument \"duration\" not specified in appropriate format. maybe try ISO format?")
    duration_sec = duration.hour * 3600 + duration.minute * 60 + duration.second + duration.microsecond / 1000000.0
    duration_frames = int(np.ceil(duration_sec / (1.0 / frame_rate)))
    # check frame2_id is within the video
    end_frame = start_frame + duration_frames
    if end_frame > total_frames:
        end_frame = total_frames

    # calculate delta frames after start_frame
    delta_frame = start_frame + args.delta
    print(f'process frame at {start_frame} and {delta_frame}')
    print(f'draw trajectory to frame {end_frame}')
    
    # (start_frame, end_frame, delta_frame are frame indices)


    # Phase 1: play the video until start frame
    frame_interval = 1.0 / frame_rate
    curr_frame = 0
    ret, frame1 = cap.read()
    curr_frame += 1
    cv2.imshow("output", frame1)
    while curr_frame < start_frame:
        ret, frame1 = cap.read()
        cv2.imshow("output", frame1)
        k = cv2.waitKey(int(frame_interval * 1000)) & 0xff
        if k == 27:
            break
        curr_frame += 1
    ret, frame2 = cap.read()
    curr_frame += 1
    cv2.imshow("output", frame2)
    while curr_frame < delta_frame:
        ret, frame2 = cap.read()
        cv2.imshow("output", frame2)
        curr_frame += 1
    # Now that frame1 and frame2 are two images for prediction.
    # frame1 is the image of start_frame
    # and frame2 the image of delta_frame
    

    # Phase 2: (MAIN TASK) predict trajectory base on frame1 and frame2

    # compute optical flow between frame1 and frame2
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # get position(s) to query for optical flow. unit is pixel
    #filtered_frame = cv2.subtract(frame1, background)
    #cv2.imshow("to_locate", filtered_frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #ball_position = locate_ball.locate_ball(filtered_frame)
    ball_position = locate_ball.locate_ball(frame1)
    assert ball_position is not None
    print('ball position:', ball_position)

    # get flow information. unit is pixel
    track_points = np.array([[ball_position]], dtype=np.float32)
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    assert track_points.shape == (1, 1, 2)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)
    track_points2, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, track_points, None, **lk_params)

    good_new = track_points2[st == 1]
    good_old = track_points[st == 1]
    if len(good_new) != 1:
        raise RuntimeError('optical flow failed to find velocity')
    old, new = good_old[0], good_new[0]

    # calculate velocity at ball_position (directly use optical flow for now),
    # then the trajectory. Note the unit of time here is frame, so unit of
    # velocity is pixel/frame.
    a, b = new.ravel()
    c, d = old.ravel()
    velocity = ((a - c) / args.delta, (b - d) / args.delta)
    print(f'velocity = {velocity}')

    # (TEMPORARY) visualize the velocity
    v = np.array(velocity)
    v = v * 10  # scale velocity for drawing
    mask = cv2.line(mask, (int(c), int(d)), (int(c+v[0]), int(d+v[1])), (0, 0, 255), 2)
    frame1 = cv2.circle(frame1, (int(c), int(d)), 5, (255,255,0), -1)

    # TODO: compute and display traj prediction
    traj_x, traj_y = compute_trajectory(ball_position, velocity, base_ng / frame_interval)
    traj_interval = end_frame - start_frame + 1
    end_x = np.floor(traj_x(traj_interval))
    start_x = np.rint(ball_position[0])
    if end_x > start_x:
        xs = np.arange(start_x, end_x+1)
    else:
        xs = np.arange(start_x, end_x-1, -1)
    ts = (xs - start_x) / velocity[0]
    ys = traj_y(ts)

    # for gravity tuning
    #tangent_vec = np.array((xs[1] - xs[0], ys[1] - ys[0]))
    #unit_v = v / np.linalg.norm(v)
    #unit_tv = tangent_vec / np.linalg.norm(tangent_vec)
    #print('tangent error =', np.linalg.norm(unit_v - unit_tv))

    xs = xs.astype(int)
    ys = np.rint(ys).astype(int)
    
    for i in range(1, xs.shape[0]):
        mask = cv2.line(mask, (xs[i-1], ys[i-1]), (xs[i], ys[i]), (0,0,255), 2)
    img = cv2.add(frame2, mask)
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    # TODO: Phase 3: keep playing video with drawn trajectory,
    # and draw ball track
    
    # draw track from frame1 to frame2
    mask = cv2.line(mask, (a,b), (c,d), (255,255,255), 2)

    p0 = good_new.reshape(-1,1,2)
    last_frame = frame2
    old_gray = frame2_gray.copy()
    while True:
        ret, frame = cap.read()
        if not ret:  # no more frames
            print("Video ended.")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # optical flow for tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]
        old, new = good_old[0], good_new[0]

        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), (255,255,255), 2)
        frame = cv2.circle(frame, (a,b), 5, (255,255,0), -1)


        img = cv2.add(frame, mask)
        cv2.imshow('output', img)
        k = cv2.waitKey(int(frame_interval * 1000)) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        p0 = good_new.reshape(-1,1,2)
        last_frame = frame
        old_gray = frame_gray.copy()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


