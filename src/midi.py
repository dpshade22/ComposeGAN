# from model import saveMidiFromCSV

# saveMidiFromCSV("bestIndividual.csv", "bestIndividual.mid")
# saveMidiFromCSV("assets/csvs/realCSVs/0.csv", "0.mid")

import os
import re

# specify the directory to search for files
directory = "./assets/realMusic"

# loop through all the directories and subdirectories in the directory tree
for root, dirs, files in os.walk(directory):
    for dir_name in dirs:
        # check if the directory name contains a dot after /pop/

        if "." in dir_name:
            dir_path = os.path.join(root, dir_name)

            # construct the new directory name by replacing dots with underscores
            new_dir_name = dir_name.replace(".", "_")

            # rename the directory to the new name
            os.rename(dir_path, new_dir_name)

            # print a message indicating that the directory has been renamed
            print(f"Renamed directory {dir_path} to {new_dir_name}")
