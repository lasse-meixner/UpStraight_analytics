# Wrapper script to automate training file building process
import sys
sys.path.append("../")

from UpStraight_Data import build_training_data

if __name__ == "__main__":
    build_training_data(path = "../data/", save = True)