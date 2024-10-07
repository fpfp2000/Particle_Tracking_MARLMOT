"""
    Perform inference on a given set of detections
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from working_test_world import TestWorld
# from train_world import TrainWorld
from dataloader_inference import TrackDataloader
from network import Net
from original_ppo import PPO
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
    parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
                        default=os.path.join(DIR_PATH, r"inference/SORT_tracks"))
    # parser.add_argument("--savepath", dest="savepath", type=str,
    #                     default=r"/Volumes/Intenso/Fernanda/Master Thesis/inference/current_tracks")
    # parser.add_argument("--savepath_2", dest="savepath_2", type=str,
    #                     default=r"/Volumes/Intenso/Fernanda/Master Thesis/inference/truth_tracks")
    # parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
    #                     default=r"/Volumes/Intenso/Fernanda/Master Thesis/inference/SORT_tracks")
    
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
                           frame_paths=frame_paths)

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
    print("Obtaining SORT batch rollouts...")

    # batch_len, \
    mota, frames, done = get_sort_rollout(dataloader, 
                            iou_threshold, 
                            min_age,
                            frame_paths)
    
    # display metrics
    # print("batch length: ", batch_len)
    print("MOTA: ", mota)

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
        color = (0, 0 ,255) if track.valid == 1 else (255, 0, 0) # red for valid == 1, blue if valid == 0

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
    savepath_SORT = args.savepath_SORT
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

########################################################################################## CHANGES MADE HERE TO GO THROUGH ALL FOLDERS 
    # for subfolder in os.listdir(datafolder):

    #     subfolder_path = os.path.join(datafolder, subfolder)

    #     # for sub in os.listdir(subfolder_path):
    #     #     subsubfolder_path = os.path.join(subfolder, sub)

    #     if not os.path.isdir(subfolder_path):
    #         continue

    #     print(f"Processing folder: {subfolder_path}")
    
    #     # dataloader = TrackDataloader(subfolder_path, mode=mode)
    #     # gt_file_path = os.path.join(subfolder_path, "gt", "gt.txt")
        

    for idx in range(len(dataloader)):
########################################################################################## ADDED GT HERE
        # get inference data
        ground_truth, detections, gt_data, gt_tracks, frame_size = dataloader.__getitem__(idx)


        if ground_truth is None or detections is None or gt_data is None:
            print(f"Skipping video {idx + 1}: Data not loaded properly")
            continue 

        # get paths to image frames
        frame_paths = dataloader.get_frame_paths(dataloader.data_paths[idx])

        # save all frames to make a video of the tracks
        # video_frames = []

    ########################################################################################## create directory to save frames
        frames_dir = os.path.join(savepath, dataloader.current_video + "_frames")
        os.makedirs(frames_dir, exist_ok=True)

        frames_dir_2 = os.path.join(savepath_2, dataloader.current_video + "_frames")
        os.makedirs(frames_dir_2, exist_ok=True)

        frames_dir_3 = os.path.join(savepath_SORT, dataloader.current_video + "_frames")
        os.makedirs(frames_dir_3, exist_ok=True)

    ########################################################################################## ADDED GT HERE
        # initialize world object to collect rollouts
        tracker = HungarianTracker(iou_threshold=iou_threshold, 
                                    min_age=min_age)
        world = TestWorld(tracker=tracker, 
                        detections=detections,
                        ground_truth=ground_truth,
                        gt_data=gt_data,
                        #   gt_tracks=gt_tracks,
                        frame_size=frame_size,
                        frame_paths=frame_paths)

        # take initial step to get first observations
        observations, _, _ = world.step({})

        # eval_sort(dataloader, iou_threshold, min_age)
        # sort_savepath = os.path.join(DIR_PATH, "sort_tracks")
        mota, done = eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT)

    ########################################################################################## MADE CHANGES HERE

        frame_count = 0
        done = False
        while not done:    

            # print(f"Processing frame {frame_count} in folder {subfolder}")

            if world.frame - 1 >= len(frame_paths):
                print(f"Frame index {world.frame - 1} out of bounds. Breaking the loop.")
                break

            frame_path = frame_paths[world.frame - 1]

            actions, logprobs = ppo.get_actions(observations)
            
            observations, _, _ = world.step(actions)

            # draw boxes on all tracks
            frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), 
                                            cv2.COLOR_BGR2RGB), 
                                            world.current_tracks)
            

            frame2 = draw_tracks_from_df(cv2.cvtColor(cv2.imread(frame_path),
                                            cv2.COLOR_BGR2RGB),
                                            world.truth_tracks)
            
            ######################################################################################
            frame3 = draw_sort_tracks(cv2.cvtColor(cv2.imread(frame_path),
                                            cv2.COLOR_BGR2RGB),
                                            world.current_tracks)
            ######################################################################################
            
            # save frame as image
            frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
            frame_filename_2 = os.path.join(frames_dir_2, f"frame_{frame_count:04d}.png")
            frame_filename_3 = os.path.join(frames_dir_3, f"frame_{frame_count:04d}.png")

            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(frame_filename_2, cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))
            cv2.imwrite(frame_filename_3, cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR))

            frame_count += 1

            # sort_savepath = os.path.join(DIR_PATH, "sort_tracks")
            # mota, done = eval_sort(dataloader, iou_threshold, min_age)

            if done:
                print("Reached end of video frames.")
                break

        # print(f"Processing of {subfolder} completed.")
        print(f"Current Tracks frames saved to: {frames_dir}")
        print(f"Truth Tracks frames saved to: {frames_dir_2}")
        print(f"SORT Tracks frames saved to: {frames_dir_3}")
    
print("ALL VIDEOS PROCESSED")
######################################################################################### CHANGES MADE HERE TO GO THROUGH ALL FOLDERS 




















# """
#     Perform inference on a given set of detections
# """

# import os
# import argparse
# import numpy as np
# import cv2
# import torch
# import matplotlib.pyplot as plt
# from working_test_world import TestWorld
# # from train_world import TrainWorld
# from dataloader_inference import TrackDataloader
# from network import Net
# from original_ppo import PPO
# from track_utils import *

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# def get_args():
#     """
#         Parses arguments from command line.
#         Outputs:
#             args - the arguments parsed
#     """
#     parser = argparse.ArgumentParser()

#     # set default path here
#     parser.add_argument("--policy", dest="policy", type=str,
#                         default=os.path.join(DIR_PATH, r"trained_models/actor_1161.pth"))
#     parser.add_argument("--datafolder", dest="datafolder", type=str, 
#                         default=r"/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/working_on_new/MOT15/train")
#     parser.add_argument("--savepath", dest="savepath", type=str,
#                         default=os.path.join(DIR_PATH, r"inference/current_tracks"))
#     parser.add_argument("--savepath_2", dest="savepath_2", type=str,
#                         default=os.path.join(DIR_PATH, r"inference/truth_tracks"))
#     parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
#                         default=os.path.join(DIR_PATH, r"inference/SORT_tracks"))
#     # parser.add_argument("--savepath", dest="savepath", type=str,
#     #                     default=r"/Volumes/Intenso/Fernanda/Master Thesis/inference/current_tracks")
#     # parser.add_argument("--savepath_2", dest="savepath_2", type=str,
#     #                     default=r"/Volumes/Intenso/Fernanda/Master Thesis/inference/truth_tracks")
#     # parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
#     #                     default=r"/Volumes/Intenso/Fernanda/Master Thesis/inference/SORT_tracks")
    
#     parser.add_argument("--idx", dest="idx", type=int, default=0)
#     parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.3)
#     parser.add_argument("--min_age", dest="min_age", type=int, default=1)
#     parser.add_argument("--video", dest="video", type=bool, choices=[True, False], default=True)
#     parser.add_argument("--mode", dest="mode", type=str, choices=["train", "test"], default="train")
#     parser.add_argument("--device", dest="device", type=str, choices=["cuda", "cpu"], default=r"cpu") 
#     args = parser.parse_args()

#     return args

# def get_sort_actions(observations):
#         """
#             Obtains actions and logprobs from current observations for SORT policy
#             i.e. action is always equal to 3
#             Inputs:
#                 observations - (dict) maps track ids to (18x1) observation vectors
#                 device - (str) device to use
#             Outputs:
#                 actions - (dict) maps track ids to discrete actions for observations
#                 logprobs -- (tesnor) log probabilities of each action
#         """
#         # handle initial frame where no observations are made
#         if len(observations) == 0:
#             return {}, []

#         # get default SORT action
#         actions = torch.ones((len(observations),)) * 3

#         # get logprob of each SORT action
#         logprobs = torch.ones_like(actions)

#         # map track IDs to actions
#         try:
#             actions = dict(zip(observations.keys(), 
#                                actions.cpu().numpy()))
#         except TypeError:
#             # handle case for length 1 observation
#             actions = dict(zip(observations.keys(), 
#                                [actions.cpu().numpy().tolist()]))
#             logprobs = logprobs.unsqueeze(0)
        
#         return actions, logprobs


# def get_sort_rollout(dataloader, iou_threshold, min_age, frame_paths):
#     """ Shameless near copy of PPO code to compute SORT rollout """
#     batch_obs = []
#     batch_actions = []
#     batch_logprobs = []
#     batch_rewards = []

#     # store metrics
#     total_num_tracks = 0

#     frames = []

# ########################################################################################## MADE AN EDIT INSIDE FOR LOOP
#     for (ground_truth, detections, gt_data, gt_tracks, frame_size) in dataloader:
        
#         # initialize world object to collect rollouts
#         tracker = HungarianTracker(iou_threshold=iou_threshold, 
#                                    min_age=min_age)
#         world = TestWorld(tracker=tracker, 
#                            ground_truth=ground_truth, 
#                            detections=detections,
#                            gt_data=gt_data,
#                            frame_size=frame_size,
#                            frame_paths=frame_paths)

#         # initialize episode rewards list 
#         ep_rewards = []

#         # accumulate total number of tracks
#         total_num_tracks += len(ground_truth)

#         # take initial step to get first observations
#         observations, rewards, done = world.step({})

#         # collect (S, A, R) trajectory for entire video
#         while not done:    

#             # append observations first
#             batch_obs += list(observations.values())

#             # take actions
#             actions, logprobs = get_sort_actions(observations)
#             # get rewards and new observations
#             observations, rewards, done = world.step(actions)

#             # store actions and new rewards 
#             batch_rewards.append(rewards)
#             batch_actions += list(actions.values())
#             batch_logprobs += logprobs

#             # assume that tracks at each frame occur at the same time step
#             ep_rewards.append(rewards) 

#             # debug
#             # print(f"world.frame: {world.frame}, len(frame_paths): {len(world.frame_paths)}")

#             # Ensuring frames are within bounds
#             if world.frame - 1 < 0 or world.frame - 1 >= len(world.frame_paths):
#                 # print(f"Frame index {world.frame - 1} is out of range for frame_paths with length {len(world.frame_paths)}")
#                 break

#             # Draw SORT tracks on the current frame
#             frame = draw_sort_tracks(cv2.cvtColor(cv2.imread(world.frame_paths[world.frame - 1]), cv2.COLOR_BGR2RGB), world.current_tracks)
#             frames.append(frame)

#     metrics = (len(batch_obs))

#     return metrics, frames, done


# def eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT):
#     """ Special function to evaluate the results of SORT on a given dataset """
#     print("Obtaining SORT batch rollouts...")

#     # batch_len, \
#     mota, frames, done = get_sort_rollout(dataloader, 
#                             iou_threshold, 
#                             min_age,
#                             frame_paths)
    
#     # display metrics
#     # print("batch length: ", batch_len)
#     print("MOTA: ", mota)

#     # saving SORT frmes
#     frames_dir = os.path.join(savepath_SORT, dataloader.current_video + "_frames")
#     os.makedirs(frames_dir, exist_ok=True)

#     for frame_count, frame in enumerate(frames):
#         frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
#         cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     print(f"SORT Tracks frames saved to: {frames_dir}")

#     return mota, done


# def draw_sort_tracks(frame, tracks):
#     """ Draws SORT bounding boxes on frame (doesn't make copy)
#         Inputs:
#             frame - current RGB video frame
#             tracks - list of track object
#         Outputs: 
#             frame - original frame with drawn bboxes
#     """
#     for track in tracks:
#         color = (0, 255, 255)  # Yellow for SORT tracks

#         # draw bbox
#         x1, y1, x2, y2 = np.round(track.get_state()[0]).astype(int)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

#         # draw track info
#         label = f"SORT_{track.id}_{track.age}"
#         frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

#     return frame

# def draw_tracks(frame, tracks):
#     """ Draws bounding boxes on frame (doesn't make copy)
#         Inputs:
#             frame - current RGB video frame
#             tracks - list of track object
#         Outputs: 
#             frame - original frame with drawn bboxes
#         """
#     for track in tracks:

#         if track.track_mode == 0:
#             color = (255, 0, 0)
#         elif track.track_mode == 1:
#             color = (0, 255, 0)
#         elif track.track_mode == 2:
#             color = (0, 0, 255)

#         # draw bbox
#         x1, y1, x2, y2 = np.round(track.get_state()[0]).astype(int)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

#         # draw track info
#         label = f"{track.id}_{track.age}"

#         frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, 
#                             color, thickness=2)

#     return frame

# def draw_tracks_from_df(frame, tracks_df):
#     """ Draws bounding boxes on frame (doesn't make copy)
#         Inputs:
#             frame - current RGB video frame
#             tracks_df - dataframe 
#         Outputs: 
#             frame - original frame with drawn bboxes
#         """
    
#     for _, track in tracks_df.iterrows():

#         # default color is red 
#         # color = (0, 0 ,255) if track.valid == 1 else (255, 0, 0) # red for valid == 1, blue if valid == 0
#         color = (0, 0 ,255) if track.conf == 1 else (255, 0, 0) # red for valid == 1, blue if valid == 0


#         # draw bbox        
#         x1, y1 = int(track.bb_left), int(track.bb_top)
#         x2, y2 = x1 + int(track.bb_width), y1 + int(track.bb_height)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

#         # draw track info
#         # label = f"{track.id}_{track.age}"
#         label = f"ID: {track.id}"

#         frame = cv2.putText(frame, label, (x1 + 10, y1 + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, 
#                             color, thickness=2)

#     return frame
   

# if __name__ == "__main__":

#     # parse arguments
#     args = get_args()
#     policy_path = args.policy
#     datafolder = args.datafolder
#     savepath = args.savepath
#     savepath_2 = args.savepath_2
#     savepath_SORT = args.savepath_SORT
#     idx = args.idx
#     iou_threshold = args.iou_threshold
#     min_age = args.min_age
#     make_video = args.video
#     mode = args.mode
#     device = args.device 

#     # get dataloader
#     dataloader = TrackDataloader(datafolder, mode= mode)

#     # get actor/policy
#     policy = Net(input_dim=18, output_dim=5).to(device)
#     # policy.load_state_dict(torch.load(policy_path))
#     policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
#     policy.eval();

#     # get default PPO class
#     ppo = PPO(dataloader, TestWorld, Net, 
#               epochs=1, 
#               iou_threshold=iou_threshold, 
#               min_age=min_age, 
#               device=device)
              
#     # set PPO actor to current actor/policy
#     ppo.actor = policy

#     for idx in range(len(dataloader)):
# ########################################################################################## ADDED GT HERE
#         # get inference data
#         ground_truth, detections, gt_data, gt_tracks, frame_size = dataloader.__getitem__(idx)


#         if ground_truth is None or detections is None or gt_data is None:
#             print(f"Skipping video {idx + 1}: Data not loaded properly")
#             continue 

#         # get paths to image frames
#         frame_paths = dataloader.get_frame_paths(dataloader.data_paths[idx])

#         # save all frames to make a video of the tracks
#         # video_frames = []

#     ########################################################################################## create directory to save frames
#         frames_dir = os.path.join(savepath, dataloader.current_video + "_frames")
#         os.makedirs(frames_dir, exist_ok=True)

#         frames_dir_2 = os.path.join(savepath_2, dataloader.current_video + "_frames")
#         os.makedirs(frames_dir_2, exist_ok=True)

#         frames_dir_3 = os.path.join(savepath_SORT, dataloader.current_video + "_frames")
#         os.makedirs(frames_dir_3, exist_ok=True)

#     ########################################################################################## ADDED GT HERE
#         # initialize world object to collect rollouts
#         tracker = HungarianTracker(iou_threshold=iou_threshold, 
#                                     min_age=min_age)
#         world = TestWorld(tracker=tracker, 
#                         detections=detections,
#                         ground_truth=ground_truth,
#                         gt_data=gt_data,
#                         #   gt_tracks=gt_tracks,
#                         frame_size=frame_size,
#                         frame_paths=frame_paths)

#         # take initial step to get first observations
#         observations, _, _ = world.step({})

#         # eval_sort(dataloader, iou_threshold, min_age)
#         # sort_savepath = os.path.join(DIR_PATH, "sort_tracks")
#         mota, done = eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT)

#     ########################################################################################## MADE CHANGES HERE

#         frame_count = 0
#         done = False
#         while not done:    

#             # print(f"Processing frame {frame_count} in folder {subfolder}")

#             if world.frame - 1 >= len(frame_paths):
#                 print(f"Frame index {world.frame - 1} out of bounds. Breaking the loop.")
#                 break

#             frame_path = frame_paths[world.frame - 1]

#             actions, logprobs = ppo.get_actions(observations)
            
#             observations, _, _ = world.step(actions)

#             # draw boxes on all tracks
#             frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), 
#                                             cv2.COLOR_BGR2RGB), 
#                                             world.current_tracks)
            

#             frame2 = draw_tracks_from_df(cv2.cvtColor(cv2.imread(frame_path),
#                                             cv2.COLOR_BGR2RGB),
#                                             world.truth_tracks)
            
#             ######################################################################################
#             frame3 = draw_sort_tracks(cv2.cvtColor(cv2.imread(frame_path),
#                                             cv2.COLOR_BGR2RGB),
#                                             world.current_tracks)
#             ######################################################################################
            
#             # save frame as image
#             frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
#             frame_filename_2 = os.path.join(frames_dir_2, f"frame_{frame_count:04d}.png")
#             frame_filename_3 = os.path.join(frames_dir_3, f"frame_{frame_count:04d}.png")

#             cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(frame_filename_2, cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(frame_filename_3, cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR))

#             frame_count += 1

#             # sort_savepath = os.path.join(DIR_PATH, "sort_tracks")
#             # mota, done = eval_sort(dataloader, iou_threshold, min_age)

#             if done:
#                 print("Reached end of video frames.")
#                 break

#         # print(f"Processing of {subfolder} completed.")
#         print(f"Current Tracks frames saved to: {frames_dir}")
#         print(f"Truth Tracks frames saved to: {frames_dir_2}")
#         print(f"SORT Tracks frames saved to: {frames_dir_3}")
    
# print("ALL VIDEOS PROCESSED")
# ######################################################################################### CHANGES MADE HERE TO GO THROUGH ALL FOLDERS 

