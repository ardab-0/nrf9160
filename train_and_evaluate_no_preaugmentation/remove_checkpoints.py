import glob
import os
folder = "grid_search_checkpoints/"

directories = [x[0] for x in os.walk(folder)]
subdirectories = directories[1:]

for dir in subdirectories:
    checkpoint_files = glob.glob(dir+"/*.ckp")
    for checkpoint_file in checkpoint_files[:-1]:
        print("Removing ", checkpoint_file)
        os.remove(checkpoint_file)

# print(subdirectories)

