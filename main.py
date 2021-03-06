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
import VelocityUtils

# gravity in pixel/sec. need tuning for different camera Orz.
#base_ng = 0.0036  # (for rvid)
base_ng = 0.00215  # or 0.0024 (for vid)
#base_ng = 0.0014
def compute_trajectory(p, v, ng, h=0.0):
    """compute poly^2 trajectory from position and velocity
       Returns x(t), y(t)
        p: position/pose (p_x, p_y)
        v: velocity (v_x, v_y) (pixel/frame)
        ng: gravity (pixel/frame^2)
    """
    xOt = np.polynomial.polynomial.Polynomial((p[0]-(v[0] * h), v[0]))
    yOt = np.polynomial.polynomial.Polynomial((p[1]-(v[1] * h)+(h*h*ng/2), v[1]-(2*h*ng/2), ng/2))
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
    cmd_parser.add_argument("--duration", help="duration of the trajectory to predict (in ISO format)", default="00:00:03")
    cmd_parser.add_argument("--delta", type=int, choices=range(1,11), help="frame difference for optical flow", default=1)
    args = cmd_parser.parse_args()

    # Read background image.
    # If not found, preprocess video and
    # store it for later.
    vidname = args.video[:args.video.rfind('.')]
    background = cv2.imread(vidname + "_background.jpg")
    if background is None:
        background = extract_background(args.video)
        cv2.imwrite(vidname + "_background.jpg", background)

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
    end_frame = start_frame + duration_frames
    # check end_frame is within the video
    if end_frame > total_frames:
        end_frame = total_frames

    # calculate delta frames after start_frame
    delta_frame = start_frame + args.delta
    print(f'process frame at {start_frame} and {delta_frame}')
    print(f'draw trajectory to frame {end_frame}')
    # input parsing ends here
    # (start_frame, end_frame, delta_frame are frame indices)




    # Step 1. play the video until start frame
    frame_interval = 1.0 / frame_rate
    curr_frame = 0
    frame1 = None
    while curr_frame < start_frame:
        curr_frame += 1
        ret, frame1 = cap.read()
        cv2.imshow("output", frame1)
        k = cv2.waitKey(int(frame_interval * 1000)) & 0xff
        if k == 27:
            break
    assert int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == curr_frame
    cv2.imwrite("frame1.jpg", frame1)
    cv2.imshow("output", frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 2. get initial position(s) of the ball. unit is pixel
    filtered_frame = cv2.subtract(frame1, background)
    ball_position = locate_ball.locate_ball(frame1, filtered_frame, viz=True)
    assert ball_position is not None
    print('ball position:', ball_position)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)

    p0 = np.array([[ball_position]], dtype=np.float32)
    old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    required_frames = [frame1]
    while True:
        curr_frame += 1
        ret, frame = cap.read()
        if not ret:  # no more frames
            print("Video ended.")
            break
        if curr_frame <= delta_frame:
            required_frames.append(frame)
        if curr_frame == delta_frame:
            assert int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == delta_frame
            assert len(required_frames) == args.delta + 1
            # now have frames for prediction,
            # so draw trajectory before proceeding
 
            frame2 = frame
            # frame1 and frame2 are two images for prediction.
            # frame1 is the image of start_frame
            # and frame2 the image of delta_frame

            # MAIN TASK HERE:
            # get velocity at start frame. unit is pixel/frame
            angles = np.linspace(0, 2*np.pi, num=8, endpoint=False)
            dx = 10 * np.cos(angles)
            dy = 10 * np.sin(angles)
            dx += ball_position[0]
            dy += ball_position[1]
            track_points = np.stack((dx, dy)).T.tolist()
            print(track_points)
            #track_points = [ball_position]

            if args.delta == 1:
                velocity = VelocityUtils.with_LK_optical_flow(frame1, frame2, track_points)
                #velocity = VelocityUtils.with_FB_optical_flow(frame1, frame2, track_points)
                print(f'velocity = {velocity}')
            else:
                # try velocities at multiple frame
                velocities = VelocityUtils.with_LK_optical_flow_N(required_frames, track_points)
                velocity = velocities[0]


            # (TEMPORARY) visualize the first velocity
            v = np.array(velocity)
            v = v * 10  # scale velocity for drawing
            c, d = ball_position[0], ball_position[1]
            mask = cv2.line(mask, (int(c), int(d)), (int(c+v[0]), int(d+v[1])), (0, 0, 255), 2)
            frame1 = cv2.circle(frame1, (int(c), int(d)), 5, (255,255,0), -1)

            # compute traj prediction from velocity
            if args.delta == 1:
                traj_x, traj_y = compute_trajectory(ball_position, velocity, base_ng / frame_interval)
            else:
                traj_x, traj_y = compute_trajectory(ball_position, velocities[0], base_ng / frame_interval)
                pos = np.array(ball_position) + velocities[0]
                for i in range(1, len(velocities)):
                    traj_xi, traj_yi = compute_trajectory(pos, velocities[i], base_ng / frame_interval, h=i)
                    traj_x += traj_xi
                    traj_y += traj_yi
                    pos += velocities[i]
                traj_x /= len(velocities)
                traj_y /= len(velocities)


            traj_interval = end_frame - start_frame + 1
            end_x = np.floor(traj_x(traj_interval))
            start_x = np.rint(ball_position[0])
            if end_x > start_x:
                xs = np.arange(start_x, end_x+1)
            else:
                xs = np.arange(start_x, end_x-1, -1)
            ts = (xs - start_x) / velocity[0]
            ys = traj_y(ts)

            # draw prediction
            xs = xs.astype(int)
            ys = np.rint(ys).astype(int)
            for i in range(1, xs.shape[0]):
                mask = cv2.line(mask, (xs[i-1], ys[i-1]), (xs[i], ys[i]), (0,0,255), 2)
            img = cv2.add(frame2, mask)
            cv2.imshow('output', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # keep playing video with drawn prediction,
        # and also draw ball track
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # optical flow for tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **(VelocityUtils.lk_params))

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
        old_gray = frame_gray.copy()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


