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


df = pd.read_csv('Train_SexAgeTIV.csv')
df = df.drop('site', axis=1)
df.target.replace({'ADHD': 1, 'hCon':0}, inplace=True)
df_vars = df[['TIV', 'Age']]

st.title("Computer-assisted detection of ADHD: binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
  st.write("### Sample Description")
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())
  if st.checkbox("Show NA"):
    st.dataframe(df.isna().sum())
 
  # across all subjects
  # distribution of target variable
  fig = plt.figure()
  sns.countplot(x='target', data=df, hue='target', legend=False)
  plt.legend(labels = ['0 = hCon', '1 = ADHD'])
  plt.title("Distribution of groups")
  st.pyplot(fig)

  # distribution of sex
  fig = plt.figure()
  sns.countplot(x='Sex', data=df, hue='Sex', legend=False)
  plt.legend(labels = ['0 = male', '1 = female'])
  plt.title("Distribution of sex")
  st.pyplot(fig)

  # distribution of TIV, Age, and correlation between vars
  fig = sns.pairplot(df_vars, kind="reg", plot_kws={'line_kws':{'color':'red'}})
  plt.title("Distribution of TIV, Age and intercorrelations", y=2.2, loc='right')
  st.pyplot(fig)
  rr = scipy.stats.pearsonr(df.Age, df.TIV)
  st.write('Across all subjects, brain volume significantly correlates with age (R =', round(rr[0],4), ', p =', round(rr[1],4), ').')

  # Statistical differences
  st.write("### A priori group differences ")
  st.write("##### (1) Do ADHD patients and hCon differ between ages?")
  st.dataframe(df.groupby(['target']).agg({'Age':['mean', 'std', 'min', 'max']}))
  group1 = df[df['target']==1]
  group2 = df[df['target']==0]
  Ttest = ttest_ind(group1['Age'], group2['Age'])
  st.write('With T=', round(Ttest[0], 2), 'and p= ', round(Ttest[1], 3), 'ADHD patients and healthy controls do not differ in age.')
  
  st.write("##### (2) Does the Total Brain Volume differ between patients and non-patients?")
  st.dataframe(df.groupby(['target']).agg({'TIV':['mean', 'std', 'min', 'max']}))
  group1 = df[df['target']==1]
  group2 = df[df['target']==0]
  Ttest = ttest_ind(group1['TIV'], group2['TIV'])
  st.write('With T=', round(Ttest[0], 1), 'and p= ', round(Ttest[1], 3), 'ADHD patients and healthy controls do not differ in brain volume')

  st.write("##### (3) Does sex distribution differ between patients and non-patients?")
  st.dataframe(df.groupby(['target']).agg({'Sex':'count'}))
  crosstab = pd.crosstab(df['target'], df['Sex'])
  chi = stats.chi2_contingency(crosstab)
  st.write('With Chi-Square=', round(chi[0], 1), 'and p=', round(chi[1], 3), 'the sex ratio differs significantly between patients and non-patients.')





if page == pages[1]:
  st.write ('### Data Visualization')

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
  st.write('##### Image Preprocessing included normalize and reslice')
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








if page == pages[2] : 
  from img_classification import teachable_machine_classification
  from PIL import Image
  
  st.header("Model Application")
  st.text("Upload a sMRI Image for image classification as ADHD or hCon")




  