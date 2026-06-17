import os
import pandas as pd

df = pd.read_csv("embed_merged_notA_dec9.csv", low_memory=False)

df['filename'] = df['anon_dicom_path'].apply(os.path.basename)

df['density'] = pd.to_numeric(
    df['tissueden'],
    errors='coerce'
)

df = df.dropna(subset=['density'])

df['density'] = df['density'].astype(int)

dataset_root = r"C:\Test_images_conv"

all_pngs = []

for split in ['train', 'val']:

    split_path = os.path.join(dataset_root, split)

    for cls in os.listdir(split_path):

        cls_path = os.path.join(split_path, cls)

        for img in os.listdir(cls_path):

            if img.endswith(".png"):

                dcm_name = img.replace(".png", ".dcm")

                all_pngs.append({
                    "image": os.path.join(cls_path, img),
                    "filename": dcm_name
                })

png_df = pd.DataFrame(all_pngs)

merged = png_df.merge(
    df[['filename', 'density']],
    on='filename'
)

merged.to_csv(
    "density_labels_8k.csv",
    index=False
)

print(len(merged))
print(merged['density'].value_counts())