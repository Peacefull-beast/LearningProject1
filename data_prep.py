import os
import shutil


# Specify the root directory where subdirectories containing files are located
root = "E:\\try2\\data"
os.chdir(root)
main_folder_list = os.listdir(root)

    

# Function to rename files in a directory
def rename_files_in_directory(directory, folder_name):
    file_count = 1
    for root, _, files in os.walk(directory):
        for filename in files:
            old_filepath = os.path.join(root, filename)
            folder_name = os.path.basename(directory)
            new_filename = f"{folder_name}_{file_count}{os.path.splitext(filename)[1]}"
            new_filepath = os.path.join(root, new_filename)
            
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {old_filepath} -> {new_filepath}")
            file_count += 1

# Iterate through subdirectories in the root directory
for dir in main_folder_list:
    path = os.path.join(root,dir)
    folder_list = os.listdir(path)
    for folder_name in folder_list:
        folder_path = os.path.join(path,folder_name)
        rename_files_in_directory(folder_path, folder_name)


        