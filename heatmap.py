from IPython.core.display import display, HTML
from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pickle
import cv2
import glob
import time
import os

from feature_extraction import extractFeatures, configParams, extract_hog_features, extract_color_features
from sliding_window import findCars, slide_window, search_windows, draw_boxes

def add_heat(heatmap, bbox_list):

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox        
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap

def apply_heat_threshold_one_image(heatmap, threshold):
    heatmap[heatmap < threshold] = 0
    return heatmap

def apply_heat_threshold(heatmap, hot_windows):
    heatmap[heatmap <= 1 + len(hot_windows)//2] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img

def get_heat_based_bboxes(img, hot_windows, verbose=False):
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    # heat = apply_heat_threshold(heat, configParams['heat_threshold'])
    heat = apply_heat_threshold(heat, hot_windows)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    if verbose is True:
        print("sliding_windows::get_heat_based_bboxes(): no.of labels: {}".format(1+labels[1]))
    
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img, heatmap

def get_heat_based_bboxes_one_image(img, hot_windows, verbose=False):
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_heat_threshold_one_image(heat, configParams['heat_threshold'])

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    if verbose is True:
        print("sliding_windows::get_heat_based_bboxes(): no.of labels: {}".format(1+labels[1]))
    
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img, heatmap