import sys
import os
sys.path.append("../env")
import pandas as pd
import numpy as np

def get_score(submission_path = "../env/submission.csv"):
    submission = pd.read_csv(submission_path, delimiter=';')
    return submission.columns[0] > submission.columns[1]

if __name__ == "__main__":
    print(get_score())
