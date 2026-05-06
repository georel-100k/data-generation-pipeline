import gzip
import json
import glob
import pandas as pd
import os

output = {'item_id':[],'class':[],'description':[],'gemma_class':[]}

for f in glob.glob("/workspace/data/listings/metadata/*.json.gz"):
    with gzip.open(f, "rt", encoding="utf-8") as file:
        for d in file.readlines():
            data = json.loads(d)
            output['item_id'].append(data['item_id'])
            d = ''
            try:
                for i in data['item_keywords']:
                        d = i['value']
            except Exception as e:
                d = data['item_name'][0]['value']
            output['class'].append(d)
            output['description'].append(data['item_name'][0]['value'])
            output['gemma_class'].append(None)

output_df = pd.DataFrame(output)
d_path = {'filepath':[],'filename':[]}
for ds in os.listdir("/workspace/data/abo/3dmodels/original"):
     ds = f"/workspace/data/abo/3dmodels/original/{ds}"
     for f in os.listdir(ds):
         if f.endswith('.glb'):
              d_path['filepath'].append(os.path.join(ds,f))
              d_path['filename'].append(f.replace('.glb',''))
              
output_df = pd.merge(pd.DataFrame(d_path), output_df, left_on='filename', right_on='item_id', how='left')
print(f"Number of missing classes: {output_df['class'].isna().sum()}")
output_df.to_csv("/workspace/data/abo_category.csv", index=False)