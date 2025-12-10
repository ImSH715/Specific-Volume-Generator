# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:12:32 2025

@author: mdp23wc

===============================================================================
Helper functions for other scripts - DO NOT EDIT SCRIPT! 
"""

import numpy as np
import os
import SimpleITK as sitk
import csv
import pandas as pd
import scipy.spatial
import skimage
import argparse
import pathlib
import glob

#%% Directory Helper Function

#Load image directory
def load_img_dir(load_dir):
    #
    img_list = os.listdir(load_dir)
    
    print(f"Found images: {img_list}")
    #
    return img_list


#Load directory of specific patient
def load_dir_old(source_dir, folder, subfolder=False, timepoint=False, filenames=False, List_all=False):
    """
    Load images from nested patient directory. Operates on a single patient.

    Structure should be: Data_Folder / Patient IDs / Scan session / folder.

    Args:
        source_dir: Base directory where patient folders are located.
        folder: Folder name to load from each scan session.
        subfolder: Subfolder within the specified folder (optional).
        timepoint: Specific time points to load (default: loads all).
        List_all: If True, includes directories in the output; otherwise, only lists images.
        
    Returns:
        Table_out: DataFrame containing image filenames and their directory paths.
    """

    Table_out = pd.DataFrame()

    # Patient path
    patient_path = os.path.join(source_dir)
    print("--")
    if timepoint:
        print(f"Timepoint(s): {timepoint}")
        if isinstance(timepoint, (list, np.ndarray)):
            time_points = np.array(timepoint)  
        else:
            time_points = np.array([timepoint])  
    else:
        print("No specified timepoints - using all directories")
        time_points = sorted([
            d for d in os.listdir(source_dir) 
            if os.path.isdir(os.path.join(source_dir, d))
        ])
    for time_point in time_points:
        print(f"Time point - {time_point}")

        # Path to the folder for the current scan session
        time_point_path = os.path.join(patient_path, time_point, folder)
        if subfolder:
            time_point_path = os.path.join(time_point_path, subfolder)

        print(f"Path: {time_point_path}")

        # Check if the folder exists
        if os.path.exists(time_point_path):
            print(f"Found folder: {time_point_path}")

            # Load directory content
            all_files = os.listdir(time_point_path)

            # Filter out directories if List_all is False
            if not List_all:
                imgs = [f for f in all_files if os.path.isfile(os.path.join(time_point_path, f))]
            else:
                imgs = all_files  # Include both files and directories

            if not imgs:
                print(f"{folder} Directory is empty!")
                imgs = ["None"]

            # Store images and directory path
            data_to_bind = {
                'Images': [imgs],   # Store the list of images
                'Directory': [time_point_path]  # Store the directory path
            }
            output_scans = pd.DataFrame(data_to_bind)

        else:
            print(f"Scan session folder not found at {time_point}!")
            data_to_bind = {
                'Images': ["None"],
                'Directory': [time_point_path]
            }
            output_scans = pd.DataFrame(data_to_bind)

        # Append the output_scans to Table_out
        Table_out = pd.concat([Table_out, output_scans], ignore_index=True)

    return Table_out
#

#Load directory of specific patient
def load_dir(source_dir, folder, subfolder=False, timepoint=False, filenames=False, List_all=False):
    """
    Load images from nested patient directory. Operates on a single patient.

    Structure should be: Data_Folder / Patient IDs / Scan session / folder.

    Args:
        source_dir: Base directory where patient folders are located.
        folder: Folder name to load from each scan session.
        subfolder: Subfolder within the specified folder (optional).
        timepoint: Specific time points to load (default: loads all).
        List_all: If True, includes directories in the output; otherwise, only lists images.
        
    Returns:
        Table_out: DataFrame containing image filenames and their directory paths.
    """

    Table_out = pd.DataFrame()
    
    print("Loading Directory Tree...")
    print(f"Directory to search: {source_dir}")
    print(f"Specified Timepoint(s): {timepoint}")
    print(f"Specified Folder(s): {folder}")
    print(f"Specified Subfolder(s): {subfolder}")
    print(f"Specified filename(s): {filenames}")
    

    # Load Folder subfunction
    def load_folder(fdir, dirs_to_find=False):
        """
        Load single folder's contents, with optional list ["directory_name1"] or ["dir2","dir3"] for specified folders only. 
        
        Params:
            fdir = folder directory to open
            dirs_to_find = array of folder strings or a single string
            
        Returns:
            dirs_to_find_list = array of existing directory names
            None if a single directory was specified but not found
        """
        
        # Track if a single directory
        single_dir_requested = False
        
        # Specified Folders
        if dirs_to_find:
            # If list
            if isinstance(dirs_to_find, (list, np.ndarray)):
                dirs_to_find_list = np.array(dirs_to_find)
                # Check if it's a list with only one element
                if len(dirs_to_find_list) == 1:
                    single_dir_requested = True
            # If single string
            else:
                dirs_to_find_list = np.array([dirs_to_find])
                single_dir_requested = True
                
            # Check if directory exists
            if not os.path.exists(fdir):
                print(f"Error: Directory '{fdir}' does not exist.")
                return np.array([])
            #
            if not os.path.isdir(fdir):
                print("Error: Directory '{fdir}' is invalid.")
                return np.array([])
                
        # No Specified Folders - Use all available
        else:
            #if path doesnt exist
            if not os.path.exists(fdir):
                print(f"Error: Directory '{fdir}' does not exist.")
                return np.array([])
            #if directory filename is incorrect
            #
            if not os.path.isdir(fdir):
                print("Error: Directory '{fdir}' is invalid.")
                return np.array([])
            #
            dirs_to_find_list = np.array(sorted([
                d for d in os.listdir(fdir) 
                if os.path.isdir(os.path.join(fdir, d))
            ]))
        
        # Check if listed directories exist, remove ones that don't exist and print error
        existing_dirs = []
        for directory in dirs_to_find_list:
            if os.path.isdir(os.path.join(fdir, directory)):
                existing_dirs.append(directory)
            else:
                print(f"Warning: Subdirectory '{directory}' not found in '{fdir}'")
        
        # Return None if a single directory was requested but not found
        if single_dir_requested and len(existing_dirs) == 0:
            return None
        
        return np.array(existing_dirs)
    #
    def load_folder_imgs(fdir, files_to_find=False):
        """
        Find image files in a directory matching specified substring patterns.
        
        Params:
            fdir = folder directory to search
            files_to_find = array of substring patterns to search for
            
        Returns:
            matching_files = array of filenames (not full paths) that match the patterns
            None if a single pattern was specified but no matches found
        """
        # Check if directory exists
        if not os.path.exists(fdir):
            print(f"Error: Directory '{fdir}' does not exist.")
            return np.array([])
        
        # Track if we're looking for a single pattern
        single_pattern_requested = False
        
        
        print("Loading files in directory...")
        # Specified file patterns
        if files_to_find:
            # If list
            if isinstance(files_to_find, (list, np.ndarray)):
                patterns_list = np.array(files_to_find)
                # Check if it's a list with only one element
                if len(patterns_list) == 1:
                    single_pattern_requested = True
            # If single string
            else:
                patterns_list = np.array([files_to_find])
                single_pattern_requested = True
                
            # Find files matching the patterns
            matching_files = []
            for pattern in patterns_list:
                # Use glob to find files containing the substring
                pattern_matches = glob.glob(os.path.join(fdir, f"*{pattern}*"))
                # Extract just the filenames, not the full paths
                pattern_matches = [os.path.basename(f) for f in pattern_matches]
                matching_files.extend(pattern_matches)
            
            # Remove duplicates and sort
            matching_files = sorted(list(set(matching_files)))
            
        # No specified patterns - get all files
        else:
            print("No specified files, listing all...")
            matching_files = sorted(glob.glob(os.path.join(fdir, "*")))
            # Filter to keep only files (not directories)
            matching_files = [os.path.basename(f) for f in matching_files if os.path.isfile(f)]
        
        # Return None if a single pattern was requested but no matches found
        if single_pattern_requested and len(matching_files) == 0:
            print("No matching files...")
            return None
        
        #Print Files found
        print(f"Files found: {matching_files}")
        
        return matching_files
        #
        
    
    
    #Run script
    patient_path = os.path.join(source_dir)
    print("--")
    
    print("Loading Patient Time Point(s)...")
    timepoints = load_folder(fdir=patient_path,dirs_to_find=timepoint)
    #
    for timepoint in timepoints:
        #
        print(f"Time point - {timepoint}")
        time_point_path = os.path.join(patient_path, timepoint)
        #
        print("Loading specified Folder(s)...")
        Folders = load_folder(fdir=time_point_path,dirs_to_find=folder)
        #
        if not Folders.all:
            print("No folders located")
            return None
        
        #
        for Folder in Folders:
            #
            print(f"Folder - {Folder}")
            folder_path = os.path.join(time_point_path, Folder)
            #
            
            #-
            #Specified subfolders
            # Specified subfolders
            if subfolder:
                if isinstance(subfolder, (list, np.ndarray)):
                    subfolder_list = subfolder
                else:
                    subfolder_list = [subfolder]
            
                subfolders = load_folder(fdir=folder_path, dirs_to_find=subfolder_list)
                
                if subfolders is None or len(subfolders) == 0:
                    print("No valid subfolders located.")
                    continue
            
                for sub in subfolders:
                    print(f"Sub-Folder - {sub}")
                    DataFolder = os.path.join(folder_path, sub)
                    
                    imgs = load_folder_imgs(fdir=DataFolder, files_to_find=filenames)
                    if imgs is None or (isinstance(imgs, np.ndarray) and len(imgs) == 0):
                        print(f"{folder} Directory is empty!")
                        imgs = np.array(["None"])
            
                    # Create and append
                    data_to_bind = {
                        'Images': [imgs],
                        'Directory': [DataFolder]
                    }
                    output_scans = pd.DataFrame(data_to_bind)
                    Table_out = pd.concat([Table_out, output_scans], ignore_index=True)
            
            else:
                # No subfolder specified — load from folder level
                DataFolder = folder_path
                imgs = load_folder_imgs(fdir=DataFolder, files_to_find=filenames)
                
                if imgs is None or (isinstance(imgs, np.ndarray) and len(imgs) == 0):
                    print(f"{folder} Directory is empty!")
                    imgs = np.array(["None"])
            
                data_to_bind = {
                    'Images': [imgs],
                    'Directory': [DataFolder]
                }
                output_scans = pd.DataFrame(data_to_bind)
                Table_out = pd.concat([Table_out, output_scans], ignore_index=True)
            #
    return Table_out    
#


#%% Load Ventilation Dirs

# #
# tdir = r"Z:\polaris2\shared\will\Octree\TestDataset\Ventilation\150083"

# t_folder = "Reg_TLC_2_RV"
# #t_folder = "Reg_TLC_2_RV"
# #vent_b_sub_dir = "Reg_TLC_2_RV"
# #t_files = ["_medfilt_3.nii.gz"] #[".nii.gz","_medfilt_1.nii.gz","_medfilt_3.nii.gz"]
# t_files = ["_medfilt_3.nii.gz"]
# t_subfolder = ["Vent_Int","Vent_Trans"]# None
# t_timepoint = ["mrA"]

# #load_dir(source_dir=t_pat_vent_dir, folder=vent_sub_dir, subfolder=vent_ventimg_dir, timepoint=vent_time_dir, filenames=vent_imgs_strings)
# # mask_str = "RV"
# # #t_subfolder = "Vent_Int"
# # #vent_b_time_dir = "mrB"
# # output_name = "Repeatability_Analysis_"
# # out_dir = "Octree"
# #

# # vent_imgs_strings = ["_medfilt_3.nii.gz"] #[".nii.gz","_medfilt_1.nii.gz","_medfilt_3.nii.gz"]
# # mask_str = "RV"
# # vent_time_dir = ["mrA"]
# # vent_ventimg_dir = ["Vent_Int","Vent_Trans"]

tdir= r"Z:\polaris2\datasets\NOCTIL_L_CT\cases"
t_timepoint = ["mrA","mrB"]
t_folder = ["Reg_TLC_2_RV","Reg_TLC_2_FRC"]
t_subfolder = "Vent_Int"
t_files = ["_medfilt_1.nii.gz","_medfilt_3.nii.gz"]

A = load_dir(source_dir = tdir, timepoint= t_timepoint, folder = t_folder, subfolder = t_subfolder, filenames=t_files)
# #B = load_dir_old(source_dir = tdir, timepoint= t_timepoint, folder = t_folder, subfolder = t_subfolder)