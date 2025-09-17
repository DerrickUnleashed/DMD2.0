import pandas as pd

df = pd.read_csv("DMD_combined_dataset.csv")

# for i in df.columns:
#     if 'P' in i:
#         print(i)

df2 = pd.read_csv("Control.csv")
for i in df2['Control Genes']:
    if i in df:
        df.drop(columns=[i],axis=1,inplace = True)
df.drop(columns=["Unnamed: 0"],axis=1,inplace = True)

df.to_csv("DMD_combined_dataset_without_Control.csv")