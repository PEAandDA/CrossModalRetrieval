import os
def delete_folder_contents(folder_path):
    """
    Deletes all files and subdirectories within the specified folder

    Parameters:
    folder_path (str): The path to the folder

    Returns:
    None
    """
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)
    os.rmdir(folder_path)
