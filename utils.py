import os

def video_paths(data_path):
    '''
        Returns list of paths to all files within directories in a data path
    '''
    files = []
    for dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir)):
            for file in os.listdir(os.path.join(data_path, dir)):
                    files.append(os.path.join(data_path, dir, file))

    return files



