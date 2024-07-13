# project_ds_sep23
Computer assisted detection of ADHD

Project for the classification between children and adolescents with and without ADHD. The project is described in detail in the finalReport document (Project_finalReport_SusanneNeufang.pdf). 

The project_dataVisualization folder contains data and codes for Data Visualization analyses. The csv-files include Sex, Age and total intracranial volumes (TIV) of the training and test sample. The Vis_apriori_Train.ipynb includes the codes for data analyses. 

The project_3d folder contains the imaging data (nifti-files of structural MRI data), the code for the deep learning classification model (project_3d.ipynb) and the file of the saved model (after training, 3d_image_classificantion.keras) 
Additionally, the folder contains a streamlit_app for model deploy (streamlit_app.py) and its required folders (__pychache__) and files (requirements.txt, Train_SexAgeTIV.csv for dataVisualization, SampleTest.txt for upload file path).
