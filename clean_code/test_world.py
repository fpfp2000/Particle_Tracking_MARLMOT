"""
    Custom/World Environment for inference with MOT15 detections
"""

import numpy as np
import pandas as pd
# from track_utils import *
from track_utils_particles import *



class TestWorld():
    def __init__(self, tracker, detections, gt_data, frame_size, ground_truth, frame_paths):
        """ Training World for visual Tracking. The ground_truth and detections
            correspond to a single video of tracks and detections.
            Args:
                tracker - Tracker Object
                detections - DataFrame of detections at every frame
                frame_size - frame size list (num rows, num cols)

            Attributes
                frame - current frame index
                current_tracks - confirmed track dictions (id -> track object)
            """
        self.tracker = tracker   # tracker class
        self.detections = detections # DataFrame of detections for offline training
        self.gt_data = gt_data
        self.frame_size = frame_size # frame size (num_rows, num_cols)
        ########################################################################################## I MADE AN EDIT HERE
        self.frame_paths = frame_paths
        self.truth_tracks = [] # current truth tracks 
        self.truth_bboxes = [] # current truth bboxes
        self.ground_truth = ground_truth # ground truth DataFrame
        ########################################################################################## I MADE AN EDIT HERE

        self.frame = 0 # current frame index
        self.current_tracks = [] # confirmed tracks

        # ensure that the Tracker is reset
        self.tracker.reset()

    @staticmethod
    def _get_detection_bboxes(detections):
        """ Obtains Detection bboxes and confidence 
            Inputs:
                detections - DataFrame of detections at current frame
            Outputs:
                detections - (Nx5) array of detections in the form of:
                    (x1, y1, x2, y2, category, confidence)
                    (left, top, right, bottom, category, confidence)
            """
        pt1 = detections.iloc[:, 1:3].to_numpy().astype(int)
        pt2 = pt1 + detections.iloc[:, 3:5].to_numpy().round().astype(int)

        return np.hstack((pt1, 
                          pt2, 
                          np.zeros((len(pt1), 1)), # category defaults to 0 for training
                          np.c_[detections.iloc[:, 5].to_numpy()]))
    

########################################################################################## I MADE AN EDIT HERE
    @staticmethod
    def _get_gt_bboxes(gt_data):
        """ Obtains Detection bboxes and confidence 
            Inputs:
                detections - DataFrame of detections at current frame
            Outputs:
                detections - (Nx5) array of detections in the form of:
                    (x1, y1, x2, y2, category, confidence)
                    (left, top, right, bottom, category, confidence)
            """
        pt1 = gt_data.iloc[:, 1:3].to_numpy().astype(int)
        pt2 = pt1 + gt_data.iloc[:, 3:5].to_numpy().round().astype(int)

        return np.hstack((pt1, 
                          pt2, 
                          np.zeros((len(pt1), 1)), # category defaults to 0 for training
                          np.c_[gt_data.iloc[:, 5].to_numpy()]))
########################################################################################## I MADE AN EDIT HERE

######################################################################################### I MADE AN EDIT HERE
    def _update_gt_bboxes(self):
        """ Update Ground Truth bboxes """
        gt_bbox = []
        # draw ground truth on frame
        for id in self.truth_tracks.id:
            track = self.truth_tracks[self.truth_tracks.id == id]
            pt1 = track.iloc[0, 2:4].to_numpy().astype(int)
            pt2 = pt1 + track.iloc[0, 4:6].to_numpy().round().astype(int)
            gt_bbox.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        # convert to array
        self.truth_bboxes = np.array(gt_bbox)
######################################################################################### I MADE AN EDIT HERE

    def update_current_tracks(self, gt = True):
        """ Update tracks for current frame """

########################################################################################## I MADE AN EDIT HERE
        # get all detections at current frame
        detections = self.detections[self.detections.frame == self.frame]
        gt_data = self.gt_data[self.gt_data.frame == self.frame]

        if not detections.empty:
            detections = self._get_detection_bboxes(detections)
        else:
            detections = np.empty((0, 6))

        if not gt_data.empty:
            gt_data = self._get_gt_bboxes(gt_data)
        else:
            gt_data = np.empty((0, 6))


        # update/associate current tracklets from tracker
        if gt == False: 
            self.current_tracks = self.tracker.update(detections)
        else:
            self.current_tracks = self.tracker.update(gt_data)
    

        # get ground truth tracks 
        self.truth_tracks = self.ground_truth.loc[self.ground_truth.frame == self.frame, :]
        
        # increment frame number
        self.frame += 1
########################################################################################## I MADE AN EDIT HERE


    def get_observations(self):
        """ Obtains a vector of all observations for the
            current frame
            """
        # area normalization parameter
        area_norm = self.frame_size[0]*self.frame_size[1] / 4

        observations = {}
        for track in self.current_tracks:
            obs = track.observation

             # normalize obsrvations
            obs[0] /= self.frame_size[1] # xpos
            obs[1] /= self.frame_size[0] # ypos
            obs[2] /= area_norm # area
            obs[4] = sigmoid(obs[4] / self.frame_size[1]) # xvel
            obs[5] = sigmoid(obs[5] / self.frame_size[0]) # yvel
            obs[6] = sigmoid(obs[6] / area_norm) # area vel

            obs[7]  /= self.frame_size[1] # detected xpos
            obs[9]  /= area_norm # detected area

            # normalize between 0-1
            obs[16] = sigmoid(obs[16] - 3) # frames since last association
            obs[17] = sigmoid(obs[17] - 3) # hit streak

            # track ID maps to observation
            observations.update({track.id : obs})

        return observations
    
    @staticmethod
    def take_action(track, action):
        """ Take action for a single observation vector 

            NOTE: for now assume all other trackfile updates
            occur within the obstacle and tracker classes

            Inputs:
                track - track file object
                action - discrete action to take
            Outputs:
                track - updated trackfile object
            """
        # terminate track
        if action == 0:
            # reset track to inactive
            track.track_mode = 0 
            # mark track to be deleted
            track.age = -1

        # restart track with detection (handles motion model failure)
        elif action == 1:
            # reset Kalman filter with new detection
            detection = convert_x_to_bbox(track.detection[0:4]).flatten()
            track.reset_kf(detection)

            # set track to visible
            track.track_mode = 1

        # filter update with prediction and detection
        elif action == 2:
            # perform update with new detection
            detection = convert_x_to_bbox(track.detection[0:4]).flatten()
            track.update(detection)

            # set track to visible
            track.track_mode = 1

        # filter update with prediction only 
        # (track detection is unreliable)
        elif action == 3:
            # no action since prediction has already been made
            # set track to visible
            track.track_mode = 1

        # filter update with prediction only
        # track is placed in a hidden state
        elif action == 4:
            # no action since prediction has already been made
            # set track to hidden
            track.track_mode = 2

        return track


    def take_actions(self, actions):
        """ Take actions for all current tracks/observations
            Inputs:
                actions - dict mapping current track id to discrete action
            """
        updated_tracks = []

        for i in range(len(self.tracker.tracks)):
            track = self.tracker.tracks[i]
            try:
                action = actions[track.id]
                # update track within tracker
                self.tracker.tracks[i] = self.take_action(track, action)
                updated_tracks.append(self.tracker.tracks[i])
            except KeyError:
                continue

        self.current_tracks = updated_tracks


    def step(self, actions):
        """ Generate observations and rewards for a single frame
            Inputs:
                # actions - (Mx1 array) actions for previous observations
                actions - length M dict that maps track ids to discrete actions for previous observations
            Outputs:
                # observations - (Nx18 array) array of (18x1) observation vectors
                observations - length N dict that maps track ids to (18x1) observation vectors
                # rewards - (Nx1) array of rewards
                rewards - length N list of rewards
                done - (Bool) indicates whether the current video is complete
            """
        # implement actions (updates current tracks)
        if len(actions) > 0:
            self.take_actions(actions)

        done = False

        # get detections and update current tracks for each frame
        self.update_current_tracks(gt=True)

        # get observations
        observations = self.get_observations()
        rewards = {}
        # self._update_gt_bboxes()

        # subtract 1 since frame count is incremented in update_current_tracks
        # subtract 1 allows for final observations before batch loop exit
        if self.detections.frame.max() == (self.frame - 1):
            done = True

        return observations, rewards, done
    

    def reset(self):
        """ Resets everything and returns an observation """
        self.frame = 0 # current frame index
        self.current_tracks = {} # confirmed tracks
        self.truth_bboxes = []
        self.truth_tracks = []

        # ensure that the Tracker is reset
        self.tracker.reset()
        observations = {-1 : np.zeros((18, 1))}

        return observations