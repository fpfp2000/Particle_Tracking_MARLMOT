

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
from test_world import TestWorld
from train_world import TrainWorld
from dataloader import TrackDataloader
from network import Net
from ppo import PPO
from track_utils import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", dest="policy", type=str,
                        default=os.path.join(DIR_PATH, r"trained_models/actor_1498.pth"))
    parser.add_argument("--datafolder", dest="datafolder", type=str, 
                        default=r"/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/csv_modified/gp3")
    parser.add_argument("--imgfolder", dest="imgfolder", type=str, 
                        default=r"/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/images/gp3")
    
    parser.add_argument("--savepath_2", dest="savepath_2", type=str,
                        default=os.path.join(DIR_PATH, r"inference_particles/truth_tracks"))
    parser.add_argument("--savepath", dest="savepath", type=str,
                        default=os.path.join(DIR_PATH, r"inference_particles/current_tracks"))
    parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
                        default=os.path.join(DIR_PATH, r"inference_particles/SORT_tracks"))

    parser.add_argument("--idx", dest="idx", type=int, default=0)
    parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.1) #0.3)
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


def get_sort_rollout(dataloader, iou_threshold, min_age, frame_paths, datafolder, color):
    """ Shameless near copy of PPO code to compute SORT rollout """
    batch_obs = []
    batch_actions = []
    batch_logprobs = []
    batch_rewards = []

    # store metrics
    total_num_tracks = 0

    frames = []

    tracker = HungarianTracker(iou_threshold=iou_threshold, 
                                   min_age=min_age)

    for idx in range(len(dataloader)):
        ground_truth, detections, gt_data, gt_tracks, frame_size = dataloader.__getitem__(idx, datafolder, color)
        
        # initialize world object to collect rollouts
        world = TestWorld(tracker=tracker, 
                           ground_truth=ground_truth, 
                           detections=detections,
                           gt_data=gt_data,
                           frame_size=frame_size,
                           frame_paths=frame_paths)                     


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


            frame_idx = world.frame - 2       
            
            if 0 <= frame_idx < len(world.frame_paths):      
               
                detections = world.detections[world.detections.frame == (frame_idx + 500)]

                # Draw SORT tracks on the current frame
                frame = draw_sort_tracks(cv2.cvtColor(cv2.imread(world.frame_paths[frame_idx]), cv2.COLOR_BGR2RGB), world.current_tracks)
                frames.append(frame)
            
            frame_idx += 1
            # # resetting tracks for each frame
            world.current_tracks = []

        
        if world.frame - 1 == len(world.frame_paths) - 1:
            frame_idx = world.frame - 1

            last_frame_detections = world.detections[world.detections.frame == (frame_idx + 500)]
            
            if last_frame_detections.empty():
                print(f"Warning no detections for the last frame {frame_idx}")
            else:
                print(f"Last frame detections: \n{last_frame_detections}")
            
            world.current_tracks = tracker.update(last_frame_detections)
            frame = draw_sort_tracks(cv2.cvtColor(cv2.imread(world.frame_paths[frame_idx]), cv2.COLOR_BGR2RGB), world.current_tracks)
            frames.append(frame)

    metrics = (len(batch_obs))

    return metrics, frames, done

 
def eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT, datafolder, color):
    """ Special function to evaluate the results of SORT on a given dataset """

    # batch_len, \
    mota, frames, done = get_sort_rollout(dataloader, 
                            iou_threshold, 
                            min_age,
                            frame_paths,
                            datafolder,
                            color)
    
    print(frames)

    # saving SORT frmes
    frames_dir = os.path.join(savepath_SORT, dataloader.current_video + "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for frame_count, frame in enumerate(frames):
        print(frame)
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"SORT Tracks frames saved to: {frames_dir}")

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
        color = (0, 255, 255)  

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
        label = f"M: {track.id}_{track.age}"
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
    ground_truth_bboxes = pd.read_csv(txt_file, usecols=[0, 1, 2, 3, 4, 5, 6], 
                                      header=None, names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"])
    return ground_truth_bboxes

def load_ground_truth_bboxes(frame_paths, ground_truth_bboxes, frames_dir_2):
    """
    Processes frames and draws only the ground truth bounding boxes, 
    and saves the resulting images to the specified directory.

    Inputs:
        frame_paths (list): List of paths to frame images.
        ground_truth_bboxes (DataFrame): DataFrame containing ground truth bounding box data.
        frames_dir_2 (str): Directory to save ground truth frames.
    """
    #iterating through each frame in the folder
    for frame in frame_paths: 
        # grabbing last three digits of frame name 
        frame_index_str = os.path.basename(frame)[-7:-4]
        frame_index = int(frame_index_str)

        # loading image
        img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)

        #grabbing ground truth bounding box for specific frame
        frame_bboxes = ground_truth_bboxes[ground_truth_bboxes["frame"] == frame_index]

        frame_with_gt = draw_tracks_from_df(img.copy(), frame_bboxes)

        frame_filename_2 = os.path.join(frames_dir_2, f"frame_{frame_index:04d}.png")
        cv2.imwrite(frame_filename_2, cv2.cvtColor(frame_with_gt, cv2.COLOR_RGB2BGR))

    print("Processing of Ground truth frames done")


def load_marlmot_bboxes(frame_paths, frames_dir, ppo, world, device):
    """
    Saves MARLMOT bounding boxes on frames for each color.

    Inputs:
        frame_paths: List of paths to image frames.
        frames_dir: Directory to save frames with MARLMOT bounding boxes.
        ppo: The PPO object for generating actions.
        world: The environment/world object.
        device: Device for computation.
    """

    # Take the initial step to get the initial observations
    observations, _, _ = world.step({})

    frame_count = 0
    # iterating over all frames
    while frame_count < len(frame_paths):
        
        frame_path = frame_paths[world.frame - 1]

        if len(observations) >= 0:
            obs_list = list(observations.values())
            obs_tensor = torch.tensor(np.array(obs_list).squeeze(), dtype=torch.float32).to(device)
            if obs_tensor.ndimension() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # Ensures batch size dimension is present
        else:
            obs_tensor = torch.tensor([]).to(device)

        # Get MARLMOT predictions
        actions, logprobs = ppo.get_actions(observations)
        # step froward with MARLMOT actions
        observations, rewards, done = world.step(actions)

        # Draw MARLMOT tracks
        frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), 
                                            cv2.COLOR_BGR2RGB), 
                                            world.current_tracks)
        
        # Save frames
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")

        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        frame_count += 1


def load_sort_bboxes(frame_paths, frames_dir_sort, dataloader, iou_threshold, min_age, datafolder, color):
    """
    Saves SORT bounding boxes on frames for each color.

    Inputs:
        dataloader: The dataloader object for loading frame data.
        iou_threshold: IoU threshold for SORT.
        min_age: Minimum age of the tracks to be considered valid.
        frame_paths: List of paths to image frames.
        frames_dir_SORT: Directory to save frames with SORT bounding boxes.
    """

    mota, frames, _ = get_sort_rollout(dataloader, iou_threshold, min_age, frame_paths, datafolder, color)
    
    frame_count = 0
    while frame_count < len(frame_paths):

        frame_filename = os.path.join(frames_dir_sort, f"frame_{frame_count:04d}.png")
        
        if frame_count < len(frames):
            frame = frames[frame_count]
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            break

        frame_count += 1

    print(f"SORT Processing done, frames saved to: {frames_dir_sort}")
            

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

    target_particle_id = 7

    # Colors to loop through
    colors = ["black", "blue", "brown", "green", "orange", "purple", "red", "yellow"]

    # get dataloader
    dataloader = TrackDataloader(imgfolder=imgfolder, mode=mode)

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
    
    # setting PPO to current actor/policy
    ppo.actor = policy

    for color in colors:
        print(f"Processing color: {color}")    
        
        # Create folders for each color's results
        frames_dir = os.path.join(savepath, f"frames_{color}")
        frames_dir_2 = os.path.join(savepath_2, f"truth_{color}")
        frames_dir_3 = os.path.join(savepath_SORT, f"sort_{color}")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(frames_dir_2, exist_ok=True)
        os.makedirs(frames_dir_3, exist_ok=True)

        # getting paths to image frames
        frame_paths = dataloader.get_frame_paths(dataloader.img_path)

        # gathering ground truth boxes
        ground_truth_bboxes = load_ground_truth(datafolder, color)
        # gathering ground truth bounding boxes and adding them on the frames
        load_ground_truth_bboxes(frame_paths=frame_paths,
                            ground_truth_bboxes=ground_truth_bboxes,
                            frames_dir_2=frames_dir_2)
        
        # MARLMOT processing 
        for idx in range(len(dataloader.img_path)):
            # Get inference data
            ground_truth, detections, gt_data, gt_tracks, frame_size = dataloader.__getitem__(idx, datafolder, color) 

            # Initialize world object to collect rollouts
            tracker = HungarianTracker(iou_threshold=iou_threshold, 
                                       min_age=min_age)
            
            world = TestWorld(tracker=tracker, 
                              detections=detections, 
                              ground_truth=ground_truth, 
                              gt_data=gt_data, 
                              frame_size=frame_size, 
                              frame_paths=frame_paths)


            load_marlmot_bboxes(frame_paths=frame_paths,
                            frames_dir=frames_dir,
                            ppo=ppo,
                            world=world,
                            device=device)
            # print("MARLMOT Processing Done")

        # SORT processing 
        load_sort_bboxes(frame_paths, frames_dir_3, dataloader, iou_threshold,  min_age, datafolder, color)
            

        print(f"Processing of {color} completed.")
        print(f"MARLMOT Tracks frames saved to: {frames_dir}")
        print(f"Truth Tracks frames saved to: {frames_dir_2}")
        print(f"SORT tracks frames saved to: {frames_dir_3}")

    print("ALL COLORS PROCESSED")
