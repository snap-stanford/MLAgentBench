import sys
import os
sys.path.append("../env")
import pandas as pd
import numpy as np

def get_score(submission_path = "../env"):
    submission = pd.read_csv(os.path.join(submission_path, "submission.csv"), delimiter=';')
    return submission.columns[0] 

if __name__ == "__main__":
    print(get_score())
