"""
    Custom dataloader for training on MOT15 Challenge data
"""

import os
import glob
import re
import configparser
import pandas as pd
import cv2
import math
from pathlib import Path

# custom dataloader for MOT Challenge data
class TrackDataloader():
    def __init__(self, imgfolder, mode="train", track_particle_id=None):
        """ Custom dataloader for MOT Challenge data
            detection_paths is assumed to always contain matching
            paths for each truth path.
            Args:
                datafolder - (str) folder where MOT15 data is stored
                mode - (str) mode for dataloader (train or test)
            """
        self.mode = mode.lower()

        self.img_path = glob.glob(os.path.join(imgfolder, "*.jpg"))
        self.img_path.sort()


        # store current ground truth and detection folder name
        self.current_video = ""

        self.track_cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "valid"]
        self.detect_cols = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
        self.track_cols_data = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"] # gt new replaced "valid" with "conf"
        self.track_particle_id = track_particle_id

    def get_gt_tracks(self, datafolder, color):
        """ Obtains ground truth tracks DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth bounding boxes
            and ids for every frame
            Inputs:
                data_folder - train folder path
                gt_cols - Desired column names for ground truth DataFrame
            Outputs:
                ground_truth_tracks - Ground Truth Tracks DataFrame 
            """
              
        txt_file = os.path.join(datafolder, f"rods_df_{color}_modified.txt")
        
        ground_truth_tracks = pd.read_csv(txt_file, usecols=[0,1,2,3,4,5,6], header=None)

        # set default column names
        ground_truth_tracks.columns = self.track_cols

        # remove invalid ground truth tracks 
        ground_truth_tracks = ground_truth_tracks[ground_truth_tracks["valid"] == 1].drop(columns=["valid"])

        if self.track_particle_id is not None:
            ground_truth_tracks = ground_truth_tracks[ground_truth_tracks["id"] == self.track_particle_id]
        

        return ground_truth_tracks
    

    def get_gt_detections(self, datafolder, color):
        """ Obtains ground truth Detections DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth detection bounding boxes
            and confidence score for every frame. Occluded objects are not included.
            Inputs:
                data_folder - train folder path
            Outputs:
                detections - Ground Truth Tracks DataFrame 
            """
              
        txt_file = os.path.join(datafolder, f"rods_df_{color}_modified.txt")

        detections = pd.read_csv(txt_file, usecols=[0, 2, 3, 4, 5, 6], header=None)        # print(f"txt files are: {txt_files}")

        detections.columns = self.detect_cols

        # scale confidence to 0-1
        detections.conf = (detections.conf - detections.conf.min()) \
                          / (detections.conf.max() - detections.conf.min())

        return detections
    
    def get_gt_data(self, datafolder, color):
        """ Obtains ground truth Detections DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth detection bounding boxes
            and confidence score for every frame. Occluded objects are not included.
            Inputs:
                data_folder - train folder path
            Outputs:
                detections - Ground Truth Tracks DataFrame 
            """

        txt_file = os.path.join(datafolder, f"rods_df_{color}_modified.txt")
  
        gt_data = pd.read_csv(txt_file, usecols=[0,2,3,4,5,6], header=None) 
      
        gt_data.columns = self.track_cols_data

        # scale confidence to 0-1
        gt_data.conf = 1
        # (gt_data.conf - gt_data.conf.min()) \
        #                   / (gt_data.conf.max() - gt_data.conf.min())

        return gt_data

    @staticmethod
    def get_frame_size(img_path):
        """ Obtains frame size for current video 
            Inputs:
                data_folder - train folder path
            Outputs:
                frame_size (num rows, num cols)
            """

        img = cv2.imread(img_path)

        height, width, _ = img.shape
        frame_size = (height, width)

        return frame_size
    

    # @staticmethod
    # def get_frame_rate():
    #     """ Obtains frame rate for current video 
    #         Inputs:
    #             data_folder - train folder path
    #         Outputs:
    #             frame_rate
    #         """
    #     # config = configparser.ConfigParser()
    #     # config.read(os.path.join(data_folder, "seqinfo.ini"))

    #     # return int(config.get("Sequence", "frameRate"))
    

    @staticmethod
    def get_frame_paths(img_path):
        """ Returns filepaths to image frames """
        # files = glob.glob(os.path.join(data_folder, "img1/*.jpg"))
        files = img_path

        file_pattern = re.compile(r'.*?(\d+).*?')

        def get_order(file):
            match = file_pattern.match(Path(file).name)
            if not match:
                return math.inf
            return int(match.groups()[0])

        sorted_files = sorted(files, key=get_order)
        return sorted_files


    def __getitem__(self, idx, datafolder, color):
        """ Obtains data for a given index
            Inputs: 
                idx - current index
            Outputs:
                ground_truth - DataFrame of Ground Truth tracks at every frame
                detections - DataFrame of detections at every frame
                frame_size - frame size list (num rows, num cols)
            """
        
        # data_paths = self.data_paths #[idx]
        img_path = self.img_path[idx]
        
        train_name = os.path.basename(datafolder)


        if self.mode == "train":
            ground_truth = self.get_gt_tracks(datafolder, color)
        else:
            ground_truth = None
        
        
        detections = self.get_gt_detections(datafolder, color)
        gt_data = self.get_gt_data(datafolder, color)
        gt_tracks = self.get_gt_tracks(datafolder, color)
        frame_size = self.get_frame_size(img_path)
        
        # store current ground truth and video names 
        self.current_video = train_name
        

        return ground_truth, detections, gt_data, gt_tracks, frame_size
    
    def __len__(self):
        return len(self.img_path)
