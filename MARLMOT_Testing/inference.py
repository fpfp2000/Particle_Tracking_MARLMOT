# """
#     Perform inference on a given set of detections
# """

# import os
# import argparse
# import numpy as np
# import cv2
# import torch
# import matplotlib.pyplot as plt
# from test_world import TestWorld
# from dataloader import TrackDataloader
# from network import Net
# from ppo import PPO
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
#                         default=r"/Users/fpfp2/Desktop/Masters Thesis/CV_tracking/MARLMOT/MOT15/test")
#     parser.add_argument("--savepath", dest="savepath", type=str,
#                         default=os.path.join(DIR_PATH, "inference"))
#     parser.add_argument("--idx", dest="idx", type=int, default=0)
#     parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.3)
#     parser.add_argument("--min_age", dest="min_age", type=int, default=1)
#     parser.add_argument("--video", dest="video", type=bool, choices=[True, False], default=True)
#     parser.add_argument("--device", dest="device", type=str, choices=["cuda", "cpu"], default=r"cpu") 
#     args = parser.parse_args()

#     return args

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


# if __name__ == "__main__":

#     # parse arguments
#     args = get_args()
#     policy_path = args.policy
#     datafolder = args.datafolder
#     savepath = args.savepath
#     idx = args.idx
#     iou_threshold = args.iou_threshold
#     min_age = args.min_age
#     make_video = args.video
#     device = args.device 

#     # get dataloader
#     dataloader = TrackDataloader(datafolder, mode="test")

#     # get actor/policy
#     policy = Net(input_dim=18, output_dim=5).to(device)
#     # policy.load_state_dict(torch.load(policy_path))
#     policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
#     policy.eval();

#     # get default PPO class
#     ppo = PPO(dataloader, TestWorld, Net, epochs=1, 
#               iou_threshold=iou_threshold, min_age=min_age, 
#               device=device)
              
#     # set PPO actor to current actor/policy
#     ppo.actor = policy

#     # get inference data
#     _, detections, frame_size = dataloader.__getitem__(idx)

#     # get paths to image frames
#     frame_paths = dataloader.get_frame_paths(dataloader.data_paths[idx])

#     # save all frames to make a video of the tracks
#     video_frames = []

#     # initialize world object to collect rollouts
#     tracker = HungarianTracker(iou_threshold=iou_threshold, 
#                                 min_age=min_age)
#     world = TestWorld(tracker=tracker, 
#                       detections=detections,
#                       frame_size=frame_size)

#     # take initial step to get first observations
#     observations, _ = world.step({})
    
#     while True:    

#         frame_path = frame_paths[world.frame - 1]

#         actions, logprobs = ppo.get_actions(observations)
#         observations, done = world.step(actions)

#         # draw boxes on all tracks
#         frame = draw_tracks(cv2.cvtColor(cv2.imread(frame_path), 
#                                          cv2.COLOR_BGR2RGB), 
#                             world.current_tracks)
#         video_frames.append(frame)

#         if done:
#             break


#     if make_video:
#         video_path = os.path.join(savepath, 
#                                 dataloader.current_video + "_tracks.mp4")
#         print(f"Saving video to: {video_path}")

#         frame_rate = dataloader.get_frame_rate(dataloader.data_paths[idx])

#         out = cv2.VideoWriter(video_path, 
#                               cv2.VideoWriter_fourcc(*'mp4v'), 
#                               frame_rate, 
#                               frame_size[::-1])

#         for frame in video_frames:
#             out.write(frame)

#         out.release()
#         del out


#######################################################################################################################
"""
    Inference script for applying a trained policy or SORT algorithm on a folder of images.
"""
import os
import argparse
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from train_world import TrainWorld
from dataloader import ImageDataloader  # Import the modified dataloader
from network import Net
from track_utils import *

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_args():
    """
        Parses arguments from command line.
        Outputs:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    # set default paths here
    parser.add_argument("--policy", dest="policy", type=str,
                        default=os.path.join(DIR_PATH, "trained_models/actor_1161.pth"))
    parser.add_argument("--imagefolder", dest="imagefolder", type=str, 
                        default=r"/path/to/image/folder")
    parser.add_argument("--outputfolder", dest="outputfolder", type=str, 
                        default=r"/path/to/output/folder")
    parser.add_argument("--iou_threshold", dest="iou_threshold", type=float, default=0.3)
    parser.add_argument("--min_age", dest="min_age", type=int, default=1)
    parser.add_argument("--device", dest="device", type=str, default=r"cpu") 
    args = parser.parse_args()

    return args


def infer_sort_on_images(image_folder, output_folder, iou_threshold, min_age):
    """ Infer SORT algorithm on a folder of images """
    # Load images
    image_paths = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')])

    # Run SORT on each image
    for image_path in image_paths:
        # Process image
        image = cv2.imread(image_path)

        # Run SORT
        # (Replace this with your SORT inference code)
        # sorted_tracks = sort_inference(image)

        # Save output
        # (Replace this with your output saving code)
        # save_sorted_tracks(sorted_tracks, output_folder)


def infer_marlmot_on_images(image_folder, policy_path, output_folder, iou_threshold, min_age, device):
    """ Infer MARLMOT policy on a folder of images """
    # Load policy
    policy = Net(input_dim=18, output_dim=5).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device(device)))
    policy.eval()

    # Initialize PPO
    ppo = PPO(ImageDataloader(image_folder), TrainWorld, Net, epochs=1, 
              iou_threshold=iou_threshold, min_age=min_age, device=device)
    ppo.actor = policy

    # Compute batch rollouts
    batch_obs, _, _, _ = ppo.batch_rollout()

    # Save output
    # (Replace this with your output saving code)
    # save_batch_output(batch_obs, output_folder)


if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    image_folder = args.imagefolder
    policy_path = args.policy
    output_folder = args.outputfolder
    iou_threshold = args.iou_threshold
    min_age = args.min_age
    device = args.device

    # Infer based on mode
    infer_marlmot_on_images(image_folder, policy_path, output_folder, iou_threshold, min_age, device)
    # Or
    # infer_sort_on_images(image_folder, output_folder, iou_threshold, min_age)

