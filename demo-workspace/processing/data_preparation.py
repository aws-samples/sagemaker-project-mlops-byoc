
import string
import os
import glob
import re
import pandas as pd
import time
import subprocess
import argparse

punc_list = string.punctuation  # you can self define list of punctuation to remove here

def remove_punctuation(text):
    """
    This function takes strings containing self defined punctuations and returns
    strings with punctuations removed.
    Input(string): one tweet, contains punctuations in the self-defined list
    Output(string): one tweet, self-defined punctuations removed
    """
    translator = str.maketrans("", "", punc_list)
    return text.translate(translator)

def staging_data(data_dir):
    for data_type in ["train", "test"]:
        data_list = []
        for label in ["neg", "pos"]:
            data_path = os.path.join(data_dir, data_type, label)
            for files in glob.glob(data_path + '/*.txt'):
                data_id = files.split('/')[-1].replace('.txt', '')
                with open(files, 'r') as f:
                    line = f.readline()
                    line = remove_punctuation(line)
                    line = re.sub("\s+", " ", line)
                    data_list.append([data_id, line, label])
                    
        data_df = pd.DataFrame(data_list, columns=["index", "text", "label"])
        data_df["event_time"] = time.time()
        data_df["data_type"] = data_type
        #data_df.reset_index(inplace=True)
        data_df.to_csv(f'/opt/ml/processing/output/raw/{data_type}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-url", type=str, required=True)
    args, _ = parser.parse_known_args()
    
    subprocess.run(f"wget {args.raw_data_url} -O aclImdb_v1.tar.gz && tar --no-same-owner -xzf aclImdb_v1.tar.gz && rm aclImdb_v1.tar.gz", shell=True)
    
    data_dir = f"{os.getcwd()}/aclImdb"
    staging_data(data_dir)
