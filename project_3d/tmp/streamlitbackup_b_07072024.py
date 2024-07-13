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
  st.write("### project_3d ")

  import nibabel as nib
  from scipy import ndimage, stats
  
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
  
  #@st.cache_data
  def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

  def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

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

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model
  
  # Build model
  model = get_model()
  # load saved model 
  model.load_weights('project_3d_trainModel.h5')
  batch_size=8
  # model compile
  initial_learning_rate = 0.0001
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
  
  model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
    run_eagerly=True,
    )
  
  ########################################################################################

  # load test data
  # Folder "Test_3D_sample/ad" contains nifti-files of ADHD patients of an independent data set.
  test_ADHD_scan_paths = [
    os.path.join(os.getcwd(), "Test_3D_sample/ad", x)
    for x in os.listdir("Test_3D_sample/ad")
  ]
  # Folder "Test_3D_sample/hc" contains nifti-files of ADHD patients of an independent data set.
  test_hCon_scan_paths = [
    os.path.join(os.getcwd(), "Test_3D_sample/hc", x)
    for x in os.listdir("Test_3D_sample/hc")  
  ]

  test_ADHD_scans = np.array([process_scan(path) for path in test_ADHD_scan_paths])
  test_hCon_scans = np.array([process_scan(path) for path in test_hCon_scan_paths])

  test_ADHD_labels = np.array([1 for _ in range(len(test_ADHD_scans))])
  test_hCon_labels = np.array([0 for _ in range(len(test_hCon_scans))])
  x_test = np.concatenate((test_ADHD_scans, test_hCon_scans), axis=0)
  y_test = np.concatenate((test_ADHD_labels, test_hCon_labels), axis=0)
  test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = (
    test_loader.shuffle(len(x_test))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
  )
  
  test_pred = model.predict(test_dataset)
  y_pred_binary =  test_pred > 0.5
  y_true = y_test
  y_pred_prob = test_pred

  from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
  import matplotlib.pyplot as plt
  import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
   
    #Confution Matrix    
  st.write('\nConfusion Matrix\n -------------------------')    
  st.write(confusion_matrix(y_true,y_pred_binary));