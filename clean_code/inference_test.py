import os
import glob
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from test_world import TestWorld
from train_world import TrainWorld
from dataloader_particles import TrackDataloader
from network_particles import Net
from ppo_particles import PPO
from track_utils_particles import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", dest="policy", type=str,
                        default=os.path.join(DIR_PATH, r"trained_models/actor_1498.pth"))
    parser.add_argument("--datafolder", dest="datafolder", type=str, 
                        default=r"/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3")
    parser.add_argument("--imgfolder", dest="imgfolder", type=str, 
                        default=r"/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/images/gp3")
    
    parser.add_argument("--savepath_2", dest="savepath_2", type=str,
                        default=os.path.join(DIR_PATH, r"inference_particles/truth_tracks"))
    parser.add_argument("--savepath", dest="savepath", type=str,
                        default=os.path.join(DIR_PATH, r"inference_particles/current_tracks"))
    parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
                        default=os.path.join(DIR_PATH, r"inference_particles/SORT_tracks"))
    
    parser.add_argument("--idx", dest="idx", type=int, default=0)
    parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.3)
    parser.add_argument("--min_age", dest="min_age", type=int, default=1)
    parser.add_argument("--video", dest="video", type=bool, choices=[True, False], default=True)
    parser.add_argument("--mode", dest="mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--device", dest="device", type=str, choices=["cuda", "cpu"], default=r"cpu")
    args = parser.parse_args()

    return args

def get_sort_actions(observations):
        """
            Obtains actions and logprobs from current observations for SORT policy
            i.e. action is always equal to 3
            Inputs:
                observations - (dict) maps track ids to (18x1) observation vectors
                device - (str) device to use
            Outputs:
                actions - (dict) maps track ids to discrete actions for observations
                logprobs -- (tesnor) log probabilities of each action
        """
        # handle initial frame where no observations are made
        if len(observations) == 0:
            return {}, []

        # get default SORT action
        actions = torch.ones((len(observations),)) * 3

        # get logprob of each SORT action
        logprobs = torch.ones_like(actions)

        # map track IDs to actions
        try:
            actions = dict(zip(observations.keys(), 
                               actions.cpu().numpy()))
        except TypeError:
            # handle case for length 1 observation
            actions = dict(zip(observations.keys(), 
                               [actions.cpu().numpy().tolist()]))
            logprobs = logprobs.unsqueeze(0)
        
        return actions, logprobs


def get_sort_rollout(dataloader, iou_threshold, min_age, frame_paths):
    """ Shameless near copy of PPO code to compute SORT rollout """
    batch_obs = []
    batch_actions = []
    batch_logprobs = []
    batch_rewards = []

    # store metrics
    total_num_tracks = 0

    frames = []

########################################################################################## MADE AN EDIT INSIDE FOR LOOP
    for (ground_truth, detections, gt_data, gt_tracks, frame_size) in dataloader:
        
        # initialize world object to collect rollouts
        tracker = HungarianTracker(iou_threshold=iou_threshold, 
                                   min_age=min_age)
        world = TestWorld(tracker=tracker, 
                           ground_truth=ground_truth, 
                           detections=detections,
                           gt_data=gt_data,
                           frame_size=frame_size,
                           frame_paths=frame_paths
                        )

        # initialize episode rewards list 
        ep_rewards = []

        # accumulate total number of tracks
        total_num_tracks += len(ground_truth)

        # take initial step to get first observations
        observations, rewards, done = world.step({})

        # collect (S, A, R) trajectory for entire video
        while not done:    

            # append observations first
            batch_obs += list(observations.values())

            # take actions
            actions, logprobs = get_sort_actions(observations)
            # get rewards and new observations
            observations, rewards, done = world.step(actions)

            # store actions and new rewards 
            batch_rewards.append(rewards)
            batch_actions += list(actions.values())
            batch_logprobs += logprobs

            # assume that tracks at each frame occur at the same time step
            ep_rewards.append(rewards) 

            # debug
            # print(f"world.frame: {world.frame}, len(frame_paths): {len(world.frame_paths)}")

            # Ensuring frames are within bounds
            if world.frame - 1 < 0 or world.frame - 1 >= len(world.frame_paths):
                # print(f"Frame index {world.frame - 1} is out of range for frame_paths with length {len(world.frame_paths)}")
                break

            # Draw SORT tracks on the current frame
            frame = draw_sort_tracks(cv2.cvtColor(cv2.imread(world.frame_paths[world.frame - 1]), cv2.COLOR_BGR2RGB), world.current_tracks)
            frames.append(frame)

    metrics = (len(batch_obs))

    return metrics, frames, done


def eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT):
    """ Special function to evaluate the results of SORT on a given dataset """
    # print("Obtaining SORT batch rollouts...")

    # batch_len, \
    mota, frames, done = get_sort_rollout(dataloader, 
                            iou_threshold, 
                            min_age,
                            frame_paths)
    
    # display metrics
    # print("batch length: ", batch_len)
    # print("MOTA: ", mota)

    # saving SORT frmes
    frames_dir = os.path.join(savepath_SORT, dataloader.current_video + "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for frame_count, frame in enumerate(frames):
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # print(f"SORT Tracks frames saved to: {frames_dir}")

    return mota, done


def draw_sort_tracks(frame, tracks):
    """ Draws SORT bounding boxes on frame (doesn't make copy)
        Inputs:
            frame - current RGB video frame
            tracks - list of track object
        Outputs: 
            frame - original frame with drawn bboxes
    """
    for track in tracks:
        color = (0, 255, 255)  # Yellow for SORT tracks

        # draw bbox
        x1, y1, x2, y2 = np.round(track.get_state()[0]).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # draw track info
        label = f"SORT_{track.id}_{track.age}"
        frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

    return frame

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

def draw_tracks_from_df(frame, tracks_df, color=(0, 255, 0)):
    """ Draws bounding boxes on frame (doesn't make copy)
        Inputs:
            frame - current RGB video frame
            tracks_df - dataframe 
        Outputs: 
            frame - original frame with drawn bboxes
        """
    for _, track in tracks_df.iterrows():
        x1, y1 = int(track.bb_left), int(track.bb_top)
        x2, y2 = x1 + int(track.bb_width), y1 + int(track.bb_height)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"ID: {track.id}"
        frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            color, thickness=2)
    return frame

def load_ground_truth(datafolder, color):
    """ Load ground truth bounding boxes from color-specific .txt files """
    txt_file = os.path.join(datafolder, f"rods_df_{color}_modified.txt")
    ground_truth_bboxes = pd.read_csv(txt_file, usecols=[0, 2, 3, 4, 5, 6], 
                                      header=None, names=["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"])
    return ground_truth_bboxes

if __name__ == "__main__":

    # parse arguments
    args = get_args()
    policy_path = args.policy
    datafolder = args.datafolder
    imgfolder = args.imgfolder
    savepath = args.savepath
    savepath_2 = args.savepath_2
    savepath_SORT = args.savepath_SORT
    idx = args.idx
    iou_threshold = args.iou_threshold
    min_age = args.min_age
    make_video = args.video
    mode = args.mode
    device = args.device

    # Colors to loop through
    colors = ["black", "blue", "brown", "green", "orange", "purple", "red", "yellow"]

    # get dataloader
    dataloader = TrackDataloader(datafolder,imgfolder, mode=mode)

    # get actor/policy
    policy = Net(input_dim=18, output_dim=5).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
    policy.eval()

    # get default PPO class
    ppo = PPO(dataloader, TestWorld, Net, 
              epochs=1, 
              iou_threshold=iou_threshold, 
              min_age=min_age, 
              device=device)
    ppo.actor = policy

    for color in colors:
        print(f"Processing color: {color}")
        ground_truth_bboxes = load_ground_truth(datafolder, color)
        
        # Create folders for each color's results
        frames_dir = os.path.join(savepath, f"frames_{color}")
        frames_dir_2 = os.path.join(savepath_2, f"truth_{color}")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(frames_dir_2, exist_ok=True)

        for idx in range(len(dataloader.img_path)):
            # Get inference data
            ground_truth, detections, gt_data, gt_tracks, frame_size = dataloader.__getitem__(idx)

            # Get paths to image frames
            frame_paths = sorted(glob.glob(os.path.join(imgfolder, "*.jpg")))
            # dataloader.get_frame_paths(dataloader.data_paths[idx])

            # Initialize world object to collect rollouts
            tracker = HungarianTracker(iou_threshold=iou_threshold, min_age=min_age)
            world = TestWorld(tracker=tracker, detections=detections, ground_truth=ground_truth, 
                              gt_data=gt_data, frame_size=frame_size, frame_paths=frame_paths)

            # Take initial step to get first observations
            observations, _, _ = world.step({})

            mota, done = eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT)


            frame_count = 0
            # print(f"Number of frames found in frame_paths: {len(frame_paths)}")
            if len(frame_paths) == 0:
                raise ValueError("No frames found in the image folder.")

            # while True:
            while frame_count < len(frame_paths):
                
                # if world.frame - 1 >= len(frame_paths):
                    # print(f"Frame index {world.frame - 1} out of bounds. Breaking the loop.")
                    # break

                # frame_path = frame_paths[world.frame - 1]
                frame_path = frame_paths[frame_count]

                print(f"Processing frame {frame_count} from {frame_path}")

                img = cv2.imread(frame_path)
                if img is None:
                    print(f"Failed to load image at: {frame_path}")
                    continue 

                # Get MARLMOT predictions
                actions, logprobs = ppo.get_actions(observations)
                observations, _, _ = world.step(actions)

                # Draw MARLMOT tracks
                print(world.current_tracks)
                frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB), world.current_tracks)

                # Filter ground truth for current frame
                current_frame_number = world.frame
                print(f"Current frame number: {current_frame_number}")

                frame_bboxes = ground_truth_bboxes[ground_truth_bboxes["frame"] == current_frame_number]
                if frame_bboxes.empty: 
                    print(f"No ground truth bounding boxes found for frame: {current_frame_number}")
                else:
                    print(f"Found bounidn gboxes for frame: {current_frame_number}")
                    
                frame_with_gt = draw_tracks_from_df(frame.copy(), frame_bboxes, color=(0, 255, 0))

                # Save frames
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                frame_filename_2 = os.path.join(frames_dir_2, f"frame_{frame_count:04d}.png")

                cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(frame_filename_2, cv2.cvtColor(frame_with_gt, cv2.COLOR_RGB2BGR))

                frame_count += 1

                if done:
                    print("Reached end of video frames.")
                    break

        print(f"Processing of {color} completed.")
        print(f"MARLMOT Tracks frames saved to: {frames_dir}")
        print(f"Truth Tracks frames saved to: {frames_dir_2}")
    print("ALL COLORS PROCESSED")