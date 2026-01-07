import os
import glob 

def get_filepaths(directory, extension, substring=None):
    text = str(directory) + '/' + extension
    filepaths = glob.glob(text)

    if substring:
        filtered_paths = [path for path in filepaths if substring in path]
        return filtered_paths
    else:
        return filepaths

        
    