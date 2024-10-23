"""
    Custom dataloader for training on MOT15 Challenge data
"""

import os
import glob
import configparser
import pandas as pd
import cv2

# custom dataloader for MOT Challenge data
class TrackDataloader():
    def __init__(self, datafolder,imgfolder, mode="train"):
        """ Custom dataloader for MOT Challenge data
            detection_paths is assumed to always contain matching
            paths for each truth path.
            Args:
                datafolder - (str) folder where MOT15 data is stored
                mode - (str) mode for dataloader (train or test)
            """
        self.mode = mode.lower()
        
        # get data
        # train_names = next(iter(os.walk(datafolder)))[1]
########################################################################################## I MADE AN EDIT HERE
        # try:
        #     train_names = next(os.walk(datafolder))[1]
        # except StopIteration:
        #     raise ValueError(f"No subdirectories found in {datafolder}")
########################################################################################## I MADE AN EDIT HERE

        # get individual folders of each video
        # self.data_paths = []
        # for name in train_names:
        #     self.data_paths.append(os.path.join(datafolder, name))

        self.data_paths = glob.glob(os.path.join("/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3", "*.txt"))

        # "/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3"
        self.img_path = glob.glob(os.path.join(imgfolder, "*.jpg"))
        self.img_path.sort()


        # store current ground truth and detection folder name
        self.current_video = ""

        self.track_cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "valid"]
        self.detect_cols = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]
        self.track_cols_data = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"] # gt new replaced "valid" with "conf"


    def get_gt_tracks(self, data_path):
        """ Obtains ground truth tracks DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth bounding boxes
            and ids for every frame
            Inputs:
                data_folder - train folder path
                gt_cols - Desired column names for ground truth DataFrame
            Outputs:
                ground_truth_tracks - Ground Truth Tracks DataFrame 
            """
        data_path = "/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3"
        # print(f"data folder is: {data_path}")
              
        txt_files = glob.glob(os.path.join(data_path, "*.txt"))
        # print(f"txt files are: {txt_files}")

        # List to hold the DataFrames
        # dataframes = []
        
        # Iterate over each txt file and attempt to read it
        # for txt_file in txt_files:
        #     print(f"Reading file: {txt_file}")
        #     try:
        #         df = pd.read_csv(txt_file, usecols=[0, 1, 2, 3, 4, 5, 6], header=None)
        #         print(f"File {txt_file} loaded successfully with shape: {df.shape}")
        #         dataframes.append(df)
        #     except pd.errors.EmptyDataError:
        #         print(f"File {txt_file} is empty or malformed.")
        #     except Exception as e:
        #         print(f"Error loading {txt_file}: {e}")
    

        # for txt_file in txt_files:
        dataframes = [pd.read_csv(txt_file,
                                          usecols=[0,1,2,3,4,5,6], 
                                          header=None) for txt_file in txt_files]
             
        ground_truth_tracks = pd.concat(dataframes, ignore_index=True)

        # set default column names
        ground_truth_tracks.columns = self.track_cols

        # remove invalid ground truth tracks 
        ground_truth_tracks = ground_truth_tracks[ground_truth_tracks["valid"] == 1].drop(columns=["valid"])

        return ground_truth_tracks
    

    def get_gt_detections(self, data_path):
        """ Obtains ground truth Detections DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth detection bounding boxes
            and confidence score for every frame. Occluded objects are not included.
            Inputs:
                data_folder - train folder path
            Outputs:
                detections - Ground Truth Tracks DataFrame 
            """
        data_path = "/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3"
        # print(f"data folder is: {data_path}")
              
        txt_files = glob.glob(os.path.join(data_path, "*.txt"))
        # print(f"txt files are: {txt_files}")

        # List to hold the DataFrames
        # dataframes = []
        
        # Iterate over each txt file and attempt to read it
        # for txt_file in txt_files:
        #     print(f"Reading file: {txt_file}")
        #     try:
        #         df = pd.read_csv(txt_file, usecols=[0, 2, 3, 4, 5, 6], header=None)
        #         print(f"File {txt_file} loaded successfully with shape: {df.shape}")
        #         dataframes.append(df)
        #     except pd.errors.EmptyDataError:
        #         print(f"File {txt_file} is empty or malformed.")
        #     except Exception as e:
        #         print(f"Error loading {txt_file}: {e}")

        # for txt_file in txt_files:
        dataframes = [pd.read_csv(txt_file,
                                          usecols=[0,2,3,4,5,6], 
                                          header=None)for txt_file in txt_files]
             
        detections = pd.concat(dataframes, ignore_index=True)

        # detections = pd.read_csv(os.path.join(data_folder, "*.txt"),
        #     # "/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3/rods_df_black_modified.txt",
        #                                             # (os.path.join(data_folder, 
        #                                             #    , "det/det.txt"), 
        #                          usecols=[0,2,3,4,5,6], 
        #                          header=None)

        detections.columns = self.detect_cols

        # scale confidence to 0-1
        detections.conf = (detections.conf - detections.conf.min()) \
                          / (detections.conf.max() - detections.conf.min())

        return detections
    
    def get_gt_data(self, data_path):
        """ Obtains ground truth Detections DataFrame from input train folder.
            Ground Truth DataFrame contains all ground truth detection bounding boxes
            and confidence score for every frame. Occluded objects are not included.
            Inputs:
                data_folder - train folder path
            Outputs:
                detections - Ground Truth Tracks DataFrame 
            """
        data_path = "/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3"

        txt_files = glob.glob(os.path.join(data_path, "*.txt"))

        # for txt_file in txt_files:
        dataframes = [pd.read_csv(txt_file,
                                          usecols=[0,2,3,4,5,6], 
                                          header=None)for txt_file in txt_files]
             
        gt_data = pd.concat(dataframes, ignore_index=True)

        # gt_data = pd.read_csv(os.path.join(data_folder, "*.txt"),
        # # pd.read_csv("/Users/fpfp2/Desktop/Masters Thesis/Particle_Tracking_MARLMOT/Particle_Tracking/csv_modified/gp3/rods_df_black_modified.txt", 
        #                          usecols=[0,2,3,4,5,6], 
        #                          header=None)

        gt_data.columns = self.track_cols_data

        # if gt_data.iloc[:, 11].isnull().any():
        #     print("NaN values found in column 11 of ground truth data")

        # gt_data.iloc[:, 11] = gt_data.iloc[:, 11].fillna(0)
 

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
        # img = cv2.imread(os.path.join(img_folder, "gp3/0500.jpg"))
        # print(img)
        height, width, _ = img.shape
        frame_size = (height, width)

        # config = configparser.ConfigParser()
        # config.read(os.path.join(data_folder, "seqinfo.ini"))
        # frame_size = (int(config.get("Sequence", "imHeight")), # num rows 
        #               int(config.get("Sequence", "imWidth")))  # num cols
        return frame_size
    

    # @staticmethod
    # def get_frame_rate(data_folder):
    #     """ Obtains frame size for current video 
    #         Inputs:
    #             data_folder - train folder path
    #         Outputs:
    #             frame_rate
    #         """
    #     config = configparser.ConfigParser()
    #     config.read(os.path.join(data_folder, "seqinfo.ini"))

    #     return int(config.get("Sequence", "frameRate"))
    
    @staticmethod
    def get_frame_paths(data_folder):
        """ Returns filepaths to image frames """
        return glob.glob(os.path.join(data_folder, "images/gp3/*.jpg"))


    def __getitem__(self, idx):
        """ Obtains data for a given index
            Inputs: 
                idx - current index
            Outputs:
                ground_truth - DataFrame of Ground Truth tracks at every frame
                detections - DataFrame of detections at every frame
                frame_size - frame size list (num rows, num cols)
            """
        
        data_folder = self.data_paths[idx]
        img_path = self.img_path[idx]
        
        train_name = os.path.basename(data_folder)
        # ground_truth = self.get_gt_tracks(data_folder)

        # print(f"Ground truth data shape: {ground_truth.shape}")
        

        # if ground_truth.shape[0] == 0:
        #     raise ValueError("Error: ground truth data is empty")

        if self.mode == "train":
            ground_truth = self.get_gt_tracks(data_folder)
        else:
            ground_truth = None
        
        
        # print(f"Loaded ground truth data shape: {ground_truth.shape}, detection shape: {detections.shape}")

        
        # if ground_truth.empty or detections.empty: 
        #     print(f"Error data for idx {idx}")
        
        detections = self.get_gt_detections(data_folder)
        gt_data = self.get_gt_data(data_folder)
        gt_tracks = self.get_gt_tracks(data_folder)
        frame_size = self.get_frame_size(img_path)
        
        # store current ground truth and video names 
        self.current_video = train_name
        

        return ground_truth, detections, gt_data, gt_tracks, frame_size
    
    def __len__(self):
        return len(self.data_paths)


# """
#     Custom dataloader for training on MOT15 Challenge data
# """

# import os
# from glob import glob
# import configparser
# import pandas as pd
# import re
# import math
# from pathlib import Path

# # custom dataloader for MOT Challenge data
# class TrackDataloader():
#     def __init__(self, datafolder, mode="train"):
#         """ Custom dataloader for MOT Challenge data
#             detection_paths is assumed to always contain matching
#             paths for each truth path.
#             Args:
#                 datafolder - (str) folder where MOT15 data is stored
#                 mode - (str) mode for dataloader (train or test)
#             """
#         self.mode = mode.lower()
        
#         # get data
# ########################################################################################## I MADE AN EDIT HERE
#         try:
#             train_names = next(os.walk(datafolder))[1]
#         except StopIteration:
#             raise ValueError(f"No subdirectories found in {datafolder}")
# ########################################################################################## I MADE AN EDIT HERE

#         # get individual folders of each video
#         self.data_paths = []
#         for name in train_names:
#             self.data_paths.append(os.path.join(datafolder, name))

#         # print("Data paths:", self.data_paths)

#         # store current ground truth and detection folder name
#         self.current_video = ""

#         self.track_cols_og = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "valid"] # gt orig
#         self.track_cols_gt_data = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"] # gt orig repaced valid with conf 
#         self.track_cols = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"] # gt new replaced "valid" with "conf"
#         self.detect_cols = ["frame", "bb_left", "bb_top", "bb_width", "bb_height", "conf"] # det
    

#     def get_gt_tracks(self, data_folder):
#         """ Obtains ground truth tracks DataFrame from input train folder.
#             Ground Truth DataFrame contains all ground truth bounding boxes
#             and ids for every frame
#             Inputs:
#                 data_folder - train folder path
#                 gt_cols - Desired column names for ground truth DataFrame
#             Outputs:
#                 ground_truth_tracks - Ground Truth Tracks DataFrame 
#             """
#         ground_truth_tracks = pd.read_csv(os.path.join(data_folder, "gt/gt.txt"), 
#                                           usecols=[0,1,2,3,4,5,6],
#                                           header=None)
#         # set default column names
#         ground_truth_tracks.columns = self.track_cols_og

#         # remove invalid ground truth tracks 
#         # ground_truth_tracks = ground_truth_tracks[ground_truth_tracks["valid"] == 1].drop(columns=["valid"])

#         return ground_truth_tracks
    

#     def get_gt_detections(self, data_folder):
#         """ Obtains ground truth Detections DataFrame from input train folder.
#             Ground Truth DataFrame contains all ground truth detection bounding boxes
#             and confidence score for every frame. Occluded objects are not included.
#             Inputs:
#                 data_folder - train folder path
#             Outputs:
#                 detections - Ground Truth Tracks DataFrame 
#             """
#         detections = pd.read_csv(os.path.join(data_folder, "det/det.txt"), 
#                                  usecols=[0,2,3,4,5,6], 
#                                  header=None)

#         detections.columns = self.detect_cols

#         # scale confidence to 0-1
#         detections.conf = (detections.conf - detections.conf.min()) \
#                           / (detections.conf.max() - detections.conf.min())

#         return detections
    
# ########################################################################################## I MADE AN EDIT HERE
    
#     def get_gt_data(self, data_folder):
#         """ Obtains ground truth Detections DataFrame from input train folder.
#             Ground Truth DataFrame contains all ground truth detection bounding boxes
#             and confidence score for every frame. Occluded objects are not included.
#             Inputs:
#                 data_folder - train folder path
#             Outputs:
#                 detections - Ground Truth Tracks DataFrame 
#             """
#         gt_data = pd.read_csv(os.path.join(data_folder, "gt/gt.txt"), 
#                                  usecols=[0,2,3,4,5,6], 
#                                  header=None)

#         gt_data.columns = self.track_cols

#         # scale confidence to 0-1
#         gt_data.conf = 1
#         # (gt_data.conf - gt_data.conf.min()) \
#         #                   / (gt_data.conf.max() - gt_data.conf.min())

#         return gt_data
# ########################################################################################## I MADE AN EDIT HERE


#     @staticmethod
#     def get_frame_size(data_folder):
#         """ Obtains frame size for current video 
#             Inputs:
#                 data_folder - train folder path
#             Outputs:
#                 frame_size (num rows, num cols)
#             """
#         config = configparser.ConfigParser()
#         config.read(os.path.join(data_folder, "seqinfo.ini"))
#         frame_size = (int(config.get("Sequence", "imHeight")), # num rows 
#                       int(config.get("Sequence", "imWidth")))  # num cols
#         return frame_size
    

#     @staticmethod
#     def get_frame_rate(data_folder):
#         """ Obtains frame size for current video 
#             Inputs:
#                 data_folder - train folder path
#             Outputs:
#                 frame_rate
#             """
#         config = configparser.ConfigParser()
#         config.read(os.path.join(data_folder, "seqinfo.ini"))

#         return int(config.get("Sequence", "frameRate"))

# ########################################################################################## I MADE AN EDIT HERE
#     @staticmethod
#     def get_frame_paths(data_folder):
#         """ Returns filepaths to image frames """
#         files = glob(os.path.join(data_folder, "img1/*.jpg"))

#         file_pattern = re.compile(r'.*?(\d+).*?')

#         def get_order(file):
#             match = file_pattern.match(Path(file).name)
#             if not match:
#                 return math.inf
#             return int(match.groups()[0])

#         sorted_files = sorted(files, key=get_order)
#         return sorted_files


#     def __getitem__(self, idx):
#         """ Obtains data for a given index
#             Inputs: 
#                 idx - current index
#             Outputs:
#                 ground_truth - DataFrame of Ground Truth tracks at every frame
#                 detections - DataFrame of detections at every frame
#                 frame_size - frame size list (num rows, num cols)
#             """
#         data_folder = self.data_paths[idx]
#         train_name = os.path.basename(data_folder)

#         if self.mode == "train":
#             ground_truth = self.get_gt_tracks(data_folder)
#         else:
#             ground_truth = None

#         detections = self.get_gt_detections(data_folder)
#         gt_data = self.get_gt_data(data_folder)
#         gt_tracks = self.get_gt_tracks(data_folder)
#         frame_size = self.get_frame_size(data_folder)

#         # store current ground truth and video names 
#         self.current_video = train_name

#         return ground_truth, detections, gt_data, gt_tracks, frame_size
    
#     def __len__(self):
#         return len(self.data_paths)
    