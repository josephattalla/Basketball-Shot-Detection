import os

def video_paths(data_path):
    files = []
    for dir in os.listdir(data_path):
        for file in os.listdir(os.path.join(data_path, dir)):
                files.append(os.path.join(data_path, dir, file))

    return files



