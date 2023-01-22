#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import cv2
from Code import Phase1_utils
#from Code import deep_swap #draw_keypoints, swap_with_ref_image, swap_within_video
import argparse

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='Data', help='Base path of images, Default: /Data')
    Parser.add_argument('--FaceDetectionMethod', default='Classical', help='Face detection method, Classical or Deep Learning? Choose from Classical and Deep, Default:Deep')
    Parser.add_argument('--Testset', type=int, default=4, help='Test Set number, Default:4')
    Parser.add_argument('--Warp', default='TPS', help='Warping Technique - Triangulation or Thin Plate Splines, Default:TPS')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    args_deep = Args.FaceDetectionMethod
    Test_set = Args.Testset
    args_warping_technique = Args.Warp

    if args_deep == 'Classical':
        if Test_set == 1:
            Phase1_utils.swap_with_ref_image(BasePath + '/Test1.mp4', BasePath+'/Rambo.jpg', BasePath+'/Test1OutputTPS_KF.avi', args_warping_technique)
        elif Test_set == 2:
            Phase1_utils.swap_within_video(BasePath + '/Test2.mp4', BasePath+'/Test2OutputTri.avi', args_warping_technique)
        elif Test_set == 3:
            Phase1_utils.swap_with_ref_image(BasePath + '/Test3.mp4', BasePath+'/Scarlett.jpg', BasePath+'/Test3OutputTri.avi', args_warping_technique)
        elif Test_set == 4:
            Phase1_utils.swap_within_video(BasePath+'/Test4.mp4', BasePath+'/Test4OutputTPS.avi', args_warping_technique)
    
    elif args_deep == 'Deep':
        if Test_set == 1:
            deep_swap.swap_with_ref_image(BasePath + '/Test1.mp4', BasePath+'/Rambo.jpg', BasePath+"/Test1OutputPRNet.avi")
        elif Test_set == 2:
            deep_swap.swap_within_video(BasePath + '/Test2.mp4', BasePath+"/Test2OutputPRNet.avi")
        elif Test_set == 3:
            deep_swap.swap_with_ref_image(BasePath + '/Test3.mp4', BasePath+'/Scarlett.jpg', BasePath+"/Test3OutputPRNet.avi")
        elif Test_set == 4:
            deep_swap.swap_within_video(BasePath + '/Test4.mp4', BasePath+"/Test4OutputPRNet.avi")

if __name__ == '__main__':
    main()
