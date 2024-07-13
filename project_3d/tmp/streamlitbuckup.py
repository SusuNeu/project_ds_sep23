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
  st.write("### (1) Do ADHD patients and hCon differ between ages?")
  st.dataframe(df.groupby(['target']).agg({'Age':['mean', 'std', 'min', 'max']}))
  group1 = df[df['target']==1]
  group2 = df[df['target']==-1]
  Ttest = ttest_ind(group1['Age'], group2['Age'])
  st.write('With T=', round(Ttest[0], 2), 'and p= ', round(Ttest[1], 3), 'ADHD patients and healthy controls do not differ in age.')
  
  st.write("### (2) Does the Total Brain Volume differ between patients and non-patients?")
  st.dataframe(df.groupby(['target']).agg({'TIV':['mean', 'std', 'min', 'max']}))
  group1 = df[df['target']==1]
  group2 = df[df['target']==-1]
  Ttest = ttest_ind(group1['TIV'], group2['TIV'])
  st.write('With T=', round(Ttest[0], 1), 'and p= ', round(Ttest[1], 3), 'ADHD patients and healthy controls do not differ in brain volume')

  st.write("### (3) Does sex distribution differ between patients and non-patients?")
  st.dataframe(df.groupby(['target']).agg({'Sex':'count'}))
  crosstab = pd.crosstab(df['target'], df['Sex'])
  chi = stats.chi2_contingency(crosstab)
  st.write('With Chi-Square=', round(chi[0], 1), 'and p=.000, the sex ratio differs significantly between patients and non-patients.')

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
   # Folder "hCon_3D_Sample" contains nifti-files of healthy controls.
  normal_scan_paths = [
    os.path.join(os.getcwd(), "hCon_3D_Sample", x)
    for x in os.listdir("hCon_3D_Sample")
  ]
  # Folder "ADHD_3D_Sample" contains nifti-files of ADHD patients.
  abnormal_scan_paths = [
    os.path.join(os.getcwd(), "ADHD_3D_Sample", x)
    for x in os.listdir("ADHD_3D_Sample")
  ]
  #st.write("hCon scans 4 train: " + str(len(normal_scan_paths)))
  #st.write("ADHD scans 4 train: " + str(len(abnormal_scan_paths)))

  ##### Train Test split
  # Each scan is resized across height, width, and depth and rescaled.
  abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
  normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

  # Scans of patients were assigned to label 1, healthy controls to label 0
  abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
  normal_labels = np.array([0 for _ in range(len(normal_scans))])

  # Split data in the ratio 70-30 for training and validation.
  x_train = np.concatenate((abnormal_scans[:70], normal_scans[:56]), axis=0)
  y_train = np.concatenate((abnormal_labels[:70], normal_labels[:56]), axis=0)
  x_val = np.concatenate((abnormal_scans[70:], normal_scans[56:]), axis=0)
  y_val = np.concatenate((abnormal_labels[70:], normal_labels[56:]), axis=0)
  #st.write("Number of samples in train and validation are %d and %d."
  #  % (x_train.shape[0], x_val.shape[0]))

  ##### Data Augmentation
  ## definition of functions 
  def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume
  
  def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label
  
  def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label
  
  ##### load data sets
  train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

  st.write('### before preprocessing')
  data = train_loader.take(1)
  images, labels = list(data)[0]
  images = images.numpy()
  image = images[0]
  st.write("File dimensions are:", image.shape)
  st.write('File Values are: ', round(image.min(), 2), round(image.max(), 2))
  st.image(np.squeeze(image[:,30]))
  
  batch_size = 2
  # Augment the on the fly during training.
  train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
  )
  # Only rescale.
  validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
  )

  # Plot one slice
  st.write('### after preprocessing')
  data = train_dataset.take(1)
  images, labels = list(data)[0]
  images = images.numpy()
  image = images[0]
  st.write("File dimensions are:", image.shape)
  st.write('File values are: ', image.min(), round(image.max(), 2))
  st.image(np.squeeze(image[:, :, 30])) 

    












