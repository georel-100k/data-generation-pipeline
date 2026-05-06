import gzip
import json
import glob
import os

import pandas as pd
df = {'filename':[], 'categories':[],'object_name':[]}

for file in glob.glob("data/objaverse/metadata/metadata/*.json.gz"):
    with gzip.open(file, "rt", encoding="utf-8") as f:
        data = json.load(f)
    for i in data.keys():
        df['filename'].append(i)
        df['object_name'].append(data[i]['name'])
        df['categories'].append(data[i]['categories'])
        # print(data[i])
    #     break
    # break
pd.DataFrame(df).to_csv("data/objaverse/metadata.csv", index=False)