"""
    Main Training script for MARLMOT

    Can set default train paths in get_args()

    Set desired hyperparameters in the main
"""
import os
import time
import argparse
from train_world import TrainWorld
from dataloader import TrackDataloader
from network import Net
from ppo import PPO


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_args():
    """
        Parses arguments from command line.
        Outputs:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    # set default paths here
    parser.add_argument("--trainfolder", dest="trainfolder", type=str,
                        default=r"/Users/fpfp2/Desktop/Masters Thesis/CV_tracking/MARLMOT/MOT15/train") 
    parser.add_argument("--savepath", dest="savepath", type=str,
                        default=os.path.join(DIR_PATH, r"trained_models"))
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    tic = time.perf_counter()

    # parse arguments
    args = get_args()
    train_folder = args.trainfolder
    savepath = args.savepath 

    # get dataloader
    dataloader = TrackDataloader(train_folder)

    # initialize PPO class with desired hyperparameters
    ppo = PPO(dataloader, TrainWorld, Net, 
              epochs=1500,       # total number of batch+training iterations
              num_train_iters=6, # number of iterations to update policy weights
              lr=5e-5,           # learning rate for policy and critic wieghts
              gamma=0.95,        # discount factor
              eps=0.2,           # clip factor (limits how size of policy update)
              iou_threshold=0.3, # iou threshold for tracker
              min_age=2,         # min age for tracks to be valid
              device=None,       # set desired device - defaults to "cuda" if available
              checkpoint=100,    # number of epochs to save check point data
              obs_dim=18,        # dimension of observation space (continous)
              action_dim=5)      # dimension of action space (discrete)
    
    # train policy
    ppo.train(savepath)

    # save training info
    ppo.save_logger(savepath)
    ppo.save_hyperparameters(savepath)
    ppo.save_metrics(savepath)

    print(f"Total Training time: {(time.perf_counter() - tic)/60} min")