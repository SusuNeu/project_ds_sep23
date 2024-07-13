import streamlit as st

# for numerical data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy import stats
import scipy.stats

# for sMRI data
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  

import keras
from keras import layers
import nibabel as nib
import random

from scipy import ndimage, stats

### -----------------------------------------------------------
###                    Page Structure 
### -----------------------------------------------------------

df = pd.read_csv('Train_SexAgeTIV.csv')
df = df.drop('site', axis=1)
df.target.replace({'ADHD': 1, 'hCon':0}, inplace=True)
df_vars = df[['TIV', 'Age']]

st.title("Computer-assisted detection of ADHD: binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "Data Visualization", "Model Deploy"]  
page=st.sidebar.radio("Go to", pages)



### -----------------------------------------------------------
###                    Data Exploration 
### -----------------------------------------------------------

if page == pages[0] : 
  st.write("### Sample Description")
  st.write("Data Frame")
  st.dataframe(df.head(10))
  st.write(df.shape)

  st.write("Data Description")
  st.dataframe(df.describe())
  if st.checkbox("Show NA"):
    st.dataframe(df.isna().sum())
 
  st.write('Distribution of Target')
  st.write(df.target.value_counts(normalize=True))
  
  st.write('Distribution of Sex')
  st.write(df.Sex.value_counts(normalize=True))
 
  

### -----------------------------------------------------------
###                   Data Visualization 
### -----------------------------------------------------------

if page == pages[1]:
  st.write ('### A    Data Visualization of numerical Data ')

  # distribution of TIV, Age, and correlation between vars
  fig = sns.pairplot(df_vars, kind="reg", plot_kws={'line_kws':{'color':'red'}})
  plt.title("Distribution of numerical variables TIV, Age and intercorrelations", y=2.2, loc='right')
  st.pyplot(fig)
  rr = scipy.stats.pearsonr(df.Age, df.TIV)
  st.write('Across all subjects, brain volume significantly correlates with age (R =', round(rr[0],4), ', p =', round(rr[1],4), ').')

  
  # Statistical differences
  st.write("##### A priori group differences ")
  st.write("##### (1) Do ADHD patients and hCon differ between ages?")
  st.dataframe(df.groupby(['target']).agg({'Age':['mean', 'std', 'min', 'max']}))
  group1 = df[df['target']==1]
  group2 = df[df['target']==0]
  Ttest = ttest_ind(group1['Age'], group2['Age'])
  st.write('With T=', round(Ttest[0], 2), 'and p= ', round(Ttest[1], 3), 'ADHD patients and healthy controls do not differ in age.')
  
  fig = sns.catplot(data = df, x = 'target', y = 'Age', kind = 'bar', hue = 'target', palette=sns.color_palette('bright')[:2], errorbar='pi')
  plt.xlabel('group')
  plt.ylabel('Age [years]')
  plt.xticks([0,1], ['hCon', 'ADHD'])
  st.pyplot(fig)

  
  st.write("##### (2) Does the Total Brain Volume differ between patients and non-patients?")
  st.dataframe(df.groupby(['target']).agg({'TIV':['mean', 'std', 'min', 'max']}))
  group1 = df[df['target']==1]
  group2 = df[df['target']==0]
  Ttest = ttest_ind(group1['TIV'], group2['TIV'])
  st.write('With T=', round(Ttest[0], 1), 'and p= ', round(Ttest[1], 3), 'ADHD patients and healthy controls do not differ in brain volume')

  fig = sns.catplot(data = df, x = 'target', y = 'TIV', kind = 'bar', hue = 'target', palette=sns.color_palette('bright')[2:4], errorbar='pi')
  plt.xlabel('group')
  plt.ylabel('brain volume [mm3]')
  plt.xticks([0,1], ['hCon', 'ADHD'])
  st.pyplot(fig)

  st.write('##### (3) Does the sex ratio differ between patients and non-patients? ')
  st.dataframe(df.groupby(['target']).agg({'Sex':'count'}))
  crosstab = pd.crosstab(df['target'], df['Sex'])
  ADHD_males=df[(df['target']==1) & (df['Sex']==0)]
  ADHD_m= ADHD_males['ID'].count()
  ADHD_females=df[(df['target']==1) & (df['Sex']==1)]
  ADHD_f= ADHD_females['ID'].count()
  hCon_males=df[(df['target']==0) & (df['Sex']==0)]
  hCon_m= hCon_males['ID'].count()
  hCon_females=df[(df['target']==0) & (df['Sex']==1)]
  hCon_f= hCon_females['ID'].count()
  st.write('In patients sex ratio is', round(100*(ADHD_m/266), 2),'% vs.',  round(100*(ADHD_f/266), 2), 
      '%, in hCons', round(100*(hCon_m/129), 2),'% vs.', round(100*(hCon_f/129), 2), '%')
  labels = 'ADHD_males', 'ADHD_females', 'hCon_males', 'hCon_females'
  sizes = [ADHD_m, ADHD_f, hCon_m, hCon_f]
  fig, ax = plt.subplots()
  ax.pie(sizes, labels=labels)
  st.pyplot(fig)

  chi = stats.chi2_contingency(crosstab)
  st.write('With Chi-Square=', round(chi[0], 1), 'and p=', round(chi[1], 3), 'the sex ratio differs significantly between patients and non-patients.')

  st.write ('### B    Data Visualization of Imaging Data')

  ##### functions for nifit-image preprocessing
  # load nifit-files
  def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan
  
  # normalize images
  def normalize(volume):
    """Normalize the volume"""
    volume = stats.zscore(volume, axis=None)
    minV = volume.min()
    maxV = volume.max()
    # volume[volume < minV] = minV
    # volume[volume > maxV] = maxV
    volume = (volume - minV) / (maxV - minV)
    volume = volume.astype("float32")
    return volume
  
  # resize images
  def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
  
  # perform preprocessing
  def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume
  
   ##### define data paths for data load
  normal_scan_paths = [
    os.path.join(os.getcwd(), "hCon_3D_Sample", x)
    for x in os.listdir("hCon_3D_Sample")
  ]

  abnormal_scan_paths = [
    os.path.join(os.getcwd(), "ADHD_3D_Sample", x)
    for x in os.listdir("ADHD_3D_Sample")
  ]

  # Plot one slice
  st.write('Image Preprocessing included normalize and reslice ')
  st.write('##### b4 preprocessing')
  image = read_nifti_file(normal_scan_paths[0])
  st.write("image dimensions are:", image.shape)
  st.write('values are between', image.min(), 'and', round(image.max(), 2))
  st.image(np.squeeze(1/100*image[:, :, 45]),clamp=True) 
  
  st.write('##### after preprocessing')
  image = process_scan(normal_scan_paths[0])
  st.write("image dimensions are:", image.shape)
  st.write('values are between', abs(round(image.min(), 2)),'and', round(image.max(), 2))
  st.image(np.squeeze(image[:, :, 30]),clamp=True) 



### -----------------------------------------------------------
###                   Model Deploy 
### -----------------------------------------------------------

if page == pages[2]:

  # Model definition
  def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define input and output of the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model 
  
  # Build model
  model = get_model(width=128, height=128, depth=64)

  # data preprocessing
  def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan
  
  def normalize(volume):
    """Normalize the volume"""
    volume = stats.zscore(volume, axis=None)
    minV = volume.min()
    maxV = volume.max()
    # volume[volume < minV] = minV
    # volume[volume > maxV] = maxV
    volume = (volume - minV) / (maxV - minV)
    volume = volume.astype("float32")
    return volume
  
  def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
  
  def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume
  
  # Model Deploy
  st.header("Model Deploy")
  st.write("Upload an sMRI Image for image classification as ADHD or normal")

  uploaded_file = st.file_uploader("Upload the sMRI Image via file path")
  if uploaded_file is not None:
        bytes_data = str(uploaded_file.read().strip().decode('utf-8'))
        st.write(bytes_data)
        x_test = process_scan(bytes_data)
        st.write("Classifying...")

        # Load best weights.
        model.load_weights("3d_image_classification.keras")
        prediction = model.predict(np.expand_dims(x_test, axis=0))[0]
        scores = [1 - prediction[0], prediction[0]]
        class_names = ["normal", "ADHD"]
        for score, name in zip(scores, class_names):
          st.write(
            "This model is %.2f percent confident that the sMRI scan is %s"
                % ((100 * score), name)
                )
  




  