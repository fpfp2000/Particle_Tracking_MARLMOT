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
from train_world import TrainWorld
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
    parser.add_argument("--savepath_SORT", dest="savepath_SORT", type=str,
                        default=os.path.join(DIR_PATH, r"inference/SORT_tracks"))
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
            logprobs -- (tensor) log probabilities of each action
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

def eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT):
    """ Special function to evaluate the results of SORT on a given dataset """
    print("Obtaining SORT batch rollouts...")

    metrics, frames, done = get_sort_rollout(dataloader, 
                            iou_threshold, 
                            min_age,
                            frame_paths)
    
    batch_len, \
    false_positives, \
    false_negatives, \
    mismatch_errors, \
    cost_penalty, \
    mota = metrics

    # display metrics
    print("batch length: ", batch_len)
    print("false positives: ", false_positives)
    print("false negatives: ", false_negatives)
    print("mismatch errrors: ", mismatch_errors)
    print("cost penalty: ", cost_penalty)
    print("total: ", false_positives 
                     + false_negatives 
                     + mismatch_errors 
                     + cost_penalty)
    print("MOTA: ", mota)

    # saving SORT frames
    frames_dir = os.path.join(savepath_SORT, dataloader.current_video + "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for frame_count, frame in enumerate(frames):
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"SORT Tracks frames saved to: {frames_dir}")

    return {
        'mota': mota,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'mismatch_errors': mismatch_errors,
        'cost_penalty': cost_penalty
    }, done


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

    # Compute a single batch on all data for MARLMOT
    print("Obtaining Batch rollouts for MARLMOT...")
    batch_obs, _, _, _ = ppo.batch_rollout()

    # Collect MARLMOT (PPO) metrics
    false_positives_marlmot = ppo.metrics["false_positives"][0]
    false_negatives_marlmot = ppo.metrics["false_negatives"][0]
    mismatch_errors_marlmot = ppo.metrics["mismatch_errors"][0]
    mota_marlmot = ppo.metrics["mota"][0]

    # Display MARLMOT metrics
    print(f"MARLMOT - MOTA: {mota_marlmot}")
    print(f"False Positives: {false_positives_marlmot}")
    print(f"False Negatives: {false_negatives_marlmot}")
    print(f"Mismatch Errors: {mismatch_errors_marlmot}")

    # Evaluate SORT
    sort_metrics, done = eval_sort(dataloader, iou_threshold, min_age, frame_paths, savepath_SORT)

    # Compare SORT and MARLMOT metrics
    print(f"Video {idx + 1} Metrics Comparison:")
    print(f"SORT - MOTA: {sort_metrics['mota']}")
    print(f"False Positives: {sort_metrics['false_positives']}")
    print(f"False Negatives: {sort_metrics['false_negatives']}")
    print(f"Mismatch Errors: {sort_metrics['mismatch_errors']}")
    
    print(f"MARLMOT - MOTA: {mota_marlmot}")
    print(f"False Positives: {false_positives_marlmot}")
    print(f"False Negatives: {false_negatives_marlmot}")
    print(f"Mismatch Errors: {mismatch_errors_marlmot}")

    print("ALL VIDEOS PROCESSED")