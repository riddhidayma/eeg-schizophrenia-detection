🧠 SchizoScan AI – AI-Powered Insights for Mental Wellness
SchizoScan AI is a deep learning + machine learning hybrid application built with Streamlit, designed to detect signs of schizophrenia in EEG (Electroencephalogram) signals. It integrates powerful models like BERT (for feature extraction) and XGBoost (for classification), enabling precise predictions using brainwave data.

🚀 Features
Upload and extract EEG data from .zip of .edf files

Feature extraction: mean, variance, frequency entropy

Augmentation for training robustness

Deep BERT-based embedding for EEG features

XGBoost classifier for final decision

Confusion matrix and classification report display

Prediction using a new .edf file

Dataset distribution and EEG feature scatter plot

🧰 Tech Stack
Component	Used Tool / Library
Frontend	Streamlit
EEG Processing	MNE
Deep Learning	BERT (via HuggingFace) + PyTorch
Classifier	XGBoost
Visualization	Matplotlib, Seaborn
Preprocessing	scikit-learn
