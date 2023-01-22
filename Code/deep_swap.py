import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from Code.api import PRN
from Code.utils.render import render_texture
import cv2

def swap_with_ref_image(ref_img_path,src_video_path,output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prn = PRN(is_dlib = True,prefix='./Code/')
    
    # texture from another image or a processed texture
    ref_img = cv2.imread(ref_img_path)
    ref_pos_list = prn.process(ref_img)
    if ref_pos_list == []:
        print("Sorry dlib couldn't detect a face in your reference image :(")
        return
    ref_pos = ref_pos_list[0]
    ref_img = ref_img/255.
    ref_texture = cv2.remap(ref_img, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture 
    new_colors = prn.get_colors_from_texture(new_texture)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(src_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            break
    cap.release()
    cap = cv2.VideoCapture(src_video_path) 
    while cap.isOpened():
        ret, src_img = cap.read()
        if ret:    
            [h, w, _] = src_img.shape
            #-- 1. 3d reconstruction -> get texture. 
            pos_list = prn.process(src_img)
            if pos_list == []:
                out.write(src_img)
                cv2.imshow("Swapped with Reference",src_img)
                cv2.waitKey(1)
                continue
            pos = pos_list[0]
            vertices = prn.get_vertices(pos)
            src_img = src_img/255.
            texture = cv2.remap(src_img, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

            #-- 3. remap to input image.(render)
            vis_colors = np.ones((vertices.shape[0], 1))
            face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
            face_mask = np.squeeze(face_mask > 0).astype(np.float32)
                        
            new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
            new_image = src_img*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

            # Possion Editing for blending image
            vis_ind = np.argwhere(face_mask>0)
            vis_min = np.min(vis_ind, 0)
            vis_max = np.max(vis_ind, 0)
            center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
            output = cv2.seamlessClone((new_image*255).astype(np.uint8), (src_img*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
            out.write(output)
            cv2.imshow("Swapped with Reference",output)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def swap_within_video(src_video_path,output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prn = PRN(is_dlib = True,prefix='./Code/')
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(src_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            break
    cap.release()
    cap = cv2.VideoCapture(src_video_path) 
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            src_img = frame    
            [h, w, _] = src_img.shape
            #-- 1. 3d reconstruction -> get texture. 
            pos_list = prn.process(src_img)
            if len(pos_list) < 2:
                out.write(src_img)
                cv2.imshow("Swapped with Reference",src_img)
                cv2.waitKey(1)
                continue
            pos1 = pos_list[0]
            pos2 = pos_list[1]

            ref_pos = pos1
            ref_img = frame/255.
            ref_texture = cv2.remap(ref_img, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
            ref_vertices = prn.get_vertices(ref_pos)
            new_texture = ref_texture 
            new_colors = prn.get_colors_from_texture(new_texture)

            pos = pos2
            vertices = prn.get_vertices(pos)
            src_img = src_img/255.
            texture = cv2.remap(src_img, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

            #-- 3. remap to input image.(render)
            vis_colors = np.ones((vertices.shape[0], 1))
            face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
            face_mask = np.squeeze(face_mask > 0).astype(np.float32)
                        
            new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
            new_image = src_img*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

            # Possion Editing for blending image
            vis_ind = np.argwhere(face_mask>0)
            vis_min = np.min(vis_ind, 0)
            vis_max = np.max(vis_ind, 0)
            center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
            output = cv2.seamlessClone((new_image*255).astype(np.uint8), (src_img*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
            
            ref_pos = pos2
            ref_img = frame/255.
            ref_texture = cv2.remap(ref_img, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
            ref_vertices = prn.get_vertices(ref_pos)
            new_texture = ref_texture
            new_colors = prn.get_colors_from_texture(new_texture)

            pos = pos1
            vertices = prn.get_vertices(pos)
            src_img = output/255.
            texture = cv2.remap(src_img, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

            #-- 3. remap to input image.(render)
            vis_colors = np.ones((vertices.shape[0], 1))
            face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
            face_mask = np.squeeze(face_mask > 0).astype(np.float32)
                        
            new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
            new_image = src_img*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

            # Possion Editing for blending image
            vis_ind = np.argwhere(face_mask>0)
            vis_min = np.min(vis_ind, 0)
            vis_max = np.max(vis_ind, 0)
            center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
            output = cv2.seamlessClone((new_image*255).astype(np.uint8), (src_img*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

            out.write(output)
            cv2.imshow("Swapped with Reference",output)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def draw_keypoints(src_video_path,output_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("yo")
    prn = PRN(is_dlib = True,prefix='./Code/')
    print("oy")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(src_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            break
    cap.release()
    cap = cv2.VideoCapture(src_video_path)
    print("first")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            src_img = frame    
            [h, w, _] = src_img.shape
            #-- 1. 3d reconstruction -> get texture.
            print("sec")
            pos_list = prn.process(src_img)
            if len(pos_list) < 2:
                out.write(src_img)
                cv2.imshow("Swapped with Reference",src_img)
                cv2.waitKey(1)
                continue
            pos1 = pos_list[0]
            pos2 = pos_list[1]
            print("third")
            output = src_img
            kpt1 = prn.get_landmarks(pos1)
            kpt2 = prn.get_landmarks(pos2)
            for kpts in [kpt1,kpt2]:
                for kpt in kpts:
                    x = kpt[0]
                    y = kpt[1]
                    cv2.circle(output, (int(x), int(y)), 1, (255,0,0), thickness=-1)
            print("fourth")
            out.write(output)
            cv2.imshow("Swapped with Reference",output)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
