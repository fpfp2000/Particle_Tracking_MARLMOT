"""
    Perform inference on a given set of detections
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from test_world import TestWorld
from dataloader import TrackDataloader
from network import Net
from ppo import PPO
from track_utils import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_args():
    """
        Parses arguments from command line.
        Outputs:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    # set default path here
    parser.add_argument("--policy", dest="policy", type=str,
                        default=os.path.join(DIR_PATH, r"trained_models/actor_1161.pth"))
    parser.add_argument("--datafolder", dest="datafolder", type=str, 
                        default=r"/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/working_on_new/MOT15/train")
    parser.add_argument("--savepath", dest="savepath", type=str,
                        default=os.path.join(DIR_PATH, r"inference/current_tracks"))
    parser.add_argument("--savepath_2", dest="savepath_2", type=str,
                        default=os.path.join(DIR_PATH, r"inference/truth_tracks"))
    parser.add_argument("--idx", dest="idx", type=int, default=0)
    parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.3)
    parser.add_argument("--min_age", dest="min_age", type=int, default=1)
    parser.add_argument("--video", dest="video", type=bool, choices=[True, False], default=True)
    parser.add_argument("--mode", dest="mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--device", dest="device", type=str, choices=["cuda", "cpu"], default=r"cpu") 
    args = parser.parse_args()

    return args

def draw_tracks(frame, tracks):
    """ Draws bounding boxes on frame (doesn't make copy)
        Inputs:
            frame - current RGB video frame
            tracks - list of track object
        Outputs: 
            frame - original frame with drawn bboxes
        """
    for track in tracks:

        if track.track_mode == 0:
            color = (255, 0, 0)
        elif track.track_mode == 1:
            color = (0, 255, 0)
        elif track.track_mode == 2:
            color = (0, 0, 255)

        # draw bbox
        x1, y1, x2, y2 = np.round(track.get_state()[0]).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # draw track info
        label = f"{track.id}_{track.age}"

        frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            color, thickness=2)

    return frame

def draw_tracks_from_df(frame, tracks_df):
    """ Draws bounding boxes on frame (doesn't make copy)
        Inputs:
            frame - current RGB video frame
            tracks_df - dataframe 
        Outputs: 
            frame - original frame with drawn bboxes
        """
    for _, track in tracks_df.iterrows():

        # default color is red 
        color = (255, 0 ,0)

        # draw bbox        
        x1, y1 = int(track.bb_left), int(track.bb_top)
        x2, y2 = x1 + int(track.bb_width), y1 + int(track.bb_height)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # draw track info
        # label = f"{track.id}_{track.age}"
        label = f"ID: {track.id}"

        frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            color, thickness=2)

    return frame

if __name__ == "__main__":

    # parse arguments
    args = get_args()
    policy_path = args.policy
    datafolder = args.datafolder
    savepath = args.savepath
    savepath_2 = args.savepath_2
    idx = args.idx
    iou_threshold = args.iou_threshold
    min_age = args.min_age
    make_video = args.video
    mode = args.mode
    device = args.device 

    # get dataloader
    dataloader = TrackDataloader(datafolder, mode= mode)

    # get actor/policy
    policy = Net(input_dim=18, output_dim=5).to(device)
    # policy.load_state_dict(torch.load(policy_path))
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
    policy.eval();

    # get default PPO class
    ppo = PPO(dataloader, TestWorld, Net, 
              epochs=1, 
              iou_threshold=iou_threshold, 
              min_age=min_age, 
              device=device)
              
    # set PPO actor to current actor/policy
    ppo.actor = policy
    

########################################################################################## ADDED GT HERE
    # get inference data
    ground_truth, detections, gt_data, gt_tracks, frame_size = dataloader.__getitem__(idx)

    # get paths to image frames
    frame_paths = dataloader.get_frame_paths(dataloader.data_paths[idx])

    # save all frames to make a video of the tracks
    # video_frames = []

########################################################################################## create directory to save frames
    frames_dir = os.path.join(savepath, dataloader.current_video + "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames_dir_2 = os.path.join(savepath_2, dataloader.current_video + "_frames")
    os.makedirs(frames_dir_2, exist_ok=True)

########################################################################################## ADDED GT HERE
    # initialize world object to collect rollouts
    tracker = HungarianTracker(iou_threshold=iou_threshold, 
                                min_age=min_age)
    world = TestWorld(tracker=tracker, 
                      detections=detections,
                      ground_truth=ground_truth,
                      gt_data=gt_data,
                    #   gt_tracks=gt_tracks,
                      frame_size=frame_size)

    # take initial step to get first observations
    observations, _ = world.step({})

########################################################################################## MADE CHANGES HERE

frame_count = 0
while True:    

    frame_path = frame_paths[world.frame - 1]

    actions, logprobs = ppo.get_actions(observations)
    observations, done = world.step(actions)

    print(f"current_tracks:{world.current_tracks}")
    # draw boxes on all tracks
    frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), 
                                    cv2.COLOR_BGR2RGB), 
                                    world.current_tracks)
    
    
    print(f"truth_tracks:{world.truth_tracks}")

    frame2 = draw_tracks_from_df(cv2.cvtColor(cv2.imread(frame_path),
                                    cv2.COLOR_BGR2RGB),
                                    world.truth_tracks)
    
    # save frame as image
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
    frame_filename_2 = os.path.join(frames_dir_2, f"frame_{frame_count:04d}.png")

    cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(frame_filename_2, cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))

    frame_count += 1

    if done:
        break

print(f"Current Tracks frames saved to: {frames_dir}")
print(f"Truth Tracks frames saved to: {frames_dir_2}")

    # while True:    

    #     frame_path = frame_paths[world.frame - 1]

    #     actions, logprobs = ppo.get_actions(observations)
    #     observations, done = world.step(actions)

    #     # draw boxes on all tracks
    #     frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), 
    #                                      cv2.COLOR_BGR2RGB), 
    #                         world.current_tracks)
    #     video_frames.append(frame)

    #     if done:
    #         break


    # if make_video:
    #     video_path = os.path.join(savepath, 
    #                             dataloader.current_video + "_tracks.mp4")
    #     print(f"Saving video to: {video_path}")

    #     frame_rate = dataloader.get_frame_rate(dataloader.data_paths[idx])

    #     out = cv2.VideoWriter(video_path, 
    #                           cv2.VideoWriter_fourcc(*'mp4v'), 
    #                           frame_rate, 
    #                           frame_size[::-1])

    #     for frame in video_frames:
    #         out.write(frame)

    #     out.release()
    #     del out


