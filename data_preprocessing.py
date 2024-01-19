import os
import torch
import cv2
import pandas as pd
import numpy as np
import pickle


df_meta = pd.read_csv("imgs/csv/meta.csv")  # read in meta data
df_dicom = pd.read_csv("imgs/csv/dicom_info.csv")  # read in dicom data

cropped_imgs = df_dicom[df_dicom["SeriesDescription"] == "cropped images"].image_path
full_mammo = df_dicom[df_dicom["SeriesDescription"] == "full mammogram images"].image_path
roi_imgs = df_dicom[df_dicom["SeriesDescription"] == "ROI mask images"].image_path

im_dir = "imgs/jpeg"

# change directory path of images
cropped_imgs = cropped_imgs.replace("CBIS-DDSM/jpeg", im_dir, regex=True)
full_mammo = full_mammo.replace("CBIS-DDSM/jpeg", im_dir, regex=True)
roi_imgs = roi_imgs.replace("CBIS-DDSM/jpeg", im_dir, regex=True)

# organize image paths
full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()

for dicom in full_mammo:
    key = dicom.split("/")[-2]
    full_mammo_dict[key] = dicom
for dicom in cropped_imgs:
    key = dicom.split("/")[-2]
    cropped_images_dict[key] = dicom
for dicom in roi_imgs:
    key = dicom.split("/")[-2]
    roi_imgs[key] = dicom

# load the mass dataset
mass_train = pd.read_csv("imgs/csv/mass_case_description_train_set.csv")
mass_test = pd.read_csv("imgs/csv/mass_case_description_test_set.csv")


# fix image paths
def fix_image_path(data):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index, 11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index, 12] = cropped_images_dict[img_name]
    print("Image paths fixed!")


# apply to datasets
fix_image_path(mass_train)
fix_image_path(mass_test)

# rename columns
mass_train = mass_train.rename(
    columns={
        "left or right breast": "left_or_right_breast",
        "image view": "image_view",
        "abnormality id": "abnormality_id",
        "abnormality type": "abnormality_type",
        "mass shape": "mass_shape",
        "mass margins": "mass_margins",
        "image file path": "image_file_path",
        "cropped image file path": "cropped_image_file_path",
        "ROI mask file path": "ROI_mask_file_path",
    }
)

# rename columns
mass_test = mass_test.rename(
    columns={
        "left or right breast": "left_or_right_breast",
        "image view": "image_view",
        "abnormality id": "abnormality_id",
        "abnormality type": "abnormality_type",
        "mass shape": "mass_shape",
        "mass margins": "mass_margins",
        "image file path": "image_file_path",
        "cropped image file path": "cropped_image_file_path",
        "ROI mask file path": "ROI_mask_file_path",
    }
)

# fill in missing values using the backwards fill method
mass_train["mass_shape"] = mass_train["mass_shape"].fillna(method="bfill")
mass_train["mass_margins"] = mass_train["mass_margins"].fillna(method="bfill")
mass_test["mass_margins"] = mass_test["mass_margins"].fillna(method="bfill")


def image_processor(image_path, target_size):
    """Preprocess images"""
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image_array = image / 255.0
    return image_array


# Merge datasets
full_mass = pd.concat([mass_train, mass_test], axis=0)

target_size = (224, 224, 3)

# Apply preprocessor to train data
full_mass["processed_images"] = full_mass["image_file_path"].apply(
    lambda x: image_processor(x, target_size)
)  # very compute intensive

# Create a binary mapper
class_mapper = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}

X_resized = np.array(full_mass["processed_images"].tolist())
with open("X_resized.pkl", "wb") as f:
    pickle.dump(X_resized, f)

# Apply class mapper to pathology column
full_mass["labels"] = full_mass["pathology"].replace(class_mapper)


def split_data(data, split_pct):
    size = int(len(data) * (1 - split_pct))
    data, test_data = torch.utils.data.random_split(data, [size, len(data) - size])
    return data, test_data


flattened_images = [img.flatten() for img in X_resized]

data = [(img.reshape(3, 224, 224), target) for img, target in zip(X_resized, full_mass["labels"])]

data_tensor = [
    (torch.tensor(img.reshape(3, 224, 224), dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    for img, target in zip(X_resized, full_mass["labels"])
]
with open("data_tensor.pkl", "wb") as f:
    pickle.dump(data_tensor, f)

with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

with open("imgs.pkl", "wb") as f:
    pickle.dump(flattened_images, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(full_mass["labels"].values, f)
