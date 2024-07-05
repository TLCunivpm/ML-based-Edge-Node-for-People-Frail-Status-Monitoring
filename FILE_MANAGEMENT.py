import os
from sklearn.model_selection import train_test_split
import numpy as np

def list_files_containing(files_arr,substring):
    idx = []
    for i in range(0,len(files_arr)):
        filepath = files_arr[i]
        filename = os.path.basename(filepath)
        if substring in filename:
            idx.append(i)

    files_arr = np.array(files_arr)

    return files_arr[idx]

def list_files_with_extension(folder_path, file_extension):
    try:
        # Get a list of all files in the specified folder
        all_files = os.listdir(folder_path)

        # Filter files based on the extension
        filtered_files = [file for file in all_files if file.endswith(file_extension)]

        absolute_paths = [os.path.join(folder_path, filename) for filename in filtered_files]

        return absolute_paths
    except FileNotFoundError:
        return f"Folder not found: {folder_path}"

import shutil

def copy_file(source_path,destination_path):
# Copy the file from source to destination
    shutil.copyfile(source_path, destination_path)

    print("File copied successfully.")

def move_file(source, destination):
    shutil.move(source, destination)
    print(f"File moved from {source} to {destination}")

def find_indices(main_array, sub_array):
    indices = []
    for item in sub_array:
        filename_item = os.path.basename(item)
        for i, string in enumerate(main_array):
            filename_string = os.path.basename(string)
            if filename_item[:-4] == filename_string[:-4]:
                indices.append(i)
                break
    return indices

def list_all_files(directory):
    '''

    :param directory: directory containing the files
    :return: the absolute paths of all the files in the directory
    '''
    all_files_relative = os.listdir(directory)
    absolute_paths = [os.path.join(directory, filename) for filename in all_files_relative]

    return absolute_paths

def list_files_in_diff_subfolder(folder_videos_path,subfolderNAMES):
    #actions = np.array(['bt_GOOD', 'fixinghair_GOOD', 'no_action_GOOD', 'wsf_GOOD'])

    files_bt = list_files_with_extension(folder_videos_path + "\\" + subfolderNAMES[0], '.mp4')
    y_bt = np.zeros(len(files_bt))
    files_fh = list_files_with_extension(folder_videos_path + "\\" +subfolderNAMES[1], '.mp4')
    y_fh = np.ones(len(files_fh))
    files_na = list_files_with_extension(folder_videos_path + "\\" +subfolderNAMES[2], '.mp4')
    y_na = np.ones(len(files_na)) * 2
    files_wsf = list_files_with_extension(folder_videos_path + "\\" +subfolderNAMES[3] , '.mp4')
    y_wsf = np.ones(len(files_wsf)) * 3

    all_files = files_bt + files_fh + files_na + files_wsf
    all_labels = np.hstack((y_bt, y_fh, y_na, y_wsf))

    return all_files,all_labels

def SPLIT(filenames,random_numb):
    '''

    :param filenames: list of filenames containing the data to split
    :param random_numb: random seed to make the experiments repeatable
    :return: the filenames of each set
    '''
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.1, random_state=random_numb)
    train_filenames, val_filenames = train_test_split(train_filenames, test_size=1.0 / 9.0, random_state=random_numb)

    return train_filenames,val_filenames,test_filenames

def create_folder_if_not_exists(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
