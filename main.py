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
import viz
from localization import locate_ball

if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("video", help="path to video of ball movement")
    cmd_parser.add_argument("start", help="time to start drawing the trajectory and used for prediction")
    cmd_parser.add_argument("duration", help="duration of the trajectory to predict")
    cmd_parser.add_argument("--delta", type=int, choices=range(1,11), help="frame difference for optical flow", default=1)
    args = cmd_parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    start, duration = None, None

    # addtional processing to validate arguments
    try:
        start = time.fromisoformat(args.start)
    except ValueError:
        raise ValueError("argument \"start\" not specified in appropriate format. maybe try HH:MM:SS?")
    # TODO: check start in video, else raise error
    try:
        duration = time.fromisoformat(args.duration)
    except ValueError:
        raise ValueError("argument \"duration\" not specified in appropriate format. maybe try HH:MM:SS?")

    # calculate corresponding frame id at time start

    # calculate delta frames after frame1_id

    # check frame2_id is within the video

    # calculate frame id at time start+duration,
    # use last frame if video length exceeded.

    # Now that frame1 and frame2 are two images for prediction.
    
    # MAIN TASK HERE
    # get position(s) to query for optical flow. unit is pixel
    ball_position = locate_ball(frame1)

    # get flow information. unit is pixel
    flow = optical_flow(frame1, frame2)

    # calculate velocity at ball_position (directly use optical flow for now),
    # then the trajectory. Note the unit of time here is frame, so unit of
    # velocity is pixel/frame. Also for traj, use frame1_id as t=0.
    #velocity = convert from representation of flow
    traj = compute_trajectory(ball_position, velocity)
    
    # VISUALIZATION
    # Now for testing purpose, just draw traj and ball on frame1 and output the image.
    # (can use cv2.imshow() or cv2.imwrite())
    # TODO: eventually, change to draw on every frame of video after start.
    traj_points = traj(np.arange(frame2_id - frame1_id + 1))
    viz_img = viz.draw_trajectory(frame1, traj_points, ball_position)
    cv2.imshow("Trajectory Prediction", viz_img)

    # perhaps output a video??


