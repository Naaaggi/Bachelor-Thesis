import os
import shutil


# Define source and destination directories
source_dir = '/Users/xxx/Downloads/speech_dataset/en/clips'
destination_dir = '/Users/xxx/Desktop/Bachelorarbeit/code/bigclipss'


# Get all files from the source directory
files = os.listdir(source_dir)

files.sort()

# number of files
print(len(files))


# Ensure we only copy the first 40000 files
files_to_copy = files[:40000]


# Copy each file to the destination directory
for file in files_to_copy:
    src_file_path = os.path.join(source_dir, file)
    if os.path.isfile(src_file_path):  # Ensure it's a file and not a directory
        shutil.copy(src_file_path, destination_dir)


print("Copy completed.")
