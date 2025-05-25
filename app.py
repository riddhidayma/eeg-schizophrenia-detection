import os
import numpy as np
import zipfile
import random
import mne
import torch
import torch.nn as nn
import transformers
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd

# Fix randomization issues
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🧠 SchizoScan AI – AI-Powered Insights for Mental Wellness.")
st.sidebar.header("Upload EEG Dataset")

uploaded_file = st.sidebar.file_uploader("Upload EEG ZIP file", type=["zip"])

if uploaded_file:
    extract_folder = "extracted_eeg"
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    st.sidebar.success("✅ Files Extracted Successfully!")


    def load_eeg(file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data, _ = raw.get_data(return_times=True)
        return data


    def extract_features(data):
        mean_vals = np.mean(data, axis=1)
        var_vals = np.var(data, axis=1)
        psd_vals = np.log(np.abs(np.fft.fft(data, axis=1)) ** 2)
        entropy = -np.sum((psd_vals / np.sum(psd_vals, axis=1, keepdims=True)) * np.log(psd_vals + 1e-10), axis=1)
        return np.concatenate((mean_vals, var_vals, entropy), axis=0)


    file_paths = [os.path.join(extract_folder, file) for file in os.listdir(extract_folder) if file.endswith(".edf")]
    X, y = [], []
    for file in file_paths:
        data = load_eeg(file)
        features = extract_features(data)
        X.append(features)
        y.append(0 if "h" in file.lower() else 1)

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    def augment_data(X, y, num_samples=5):
        X_aug, y_aug = [], []
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.05, X.shape)
            X_aug.append(X + noise)
            y_aug.append(y)
        return np.vstack(X_aug), np.hstack(y_aug)


    X_train, y_train = augment_data(X_train, y_train)

    st.write("🟡 **Training Model... (BERT Feature Extraction in Progress)**")

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


    class BERTFeatureExtractor(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.bert = transformers.AutoModel.from_pretrained("bert-base-uncased").to(device)
            self.fc = nn.Linear(768, hidden_dim).to(device)

        def forward(self, x):
            text_data = [" ".join(map(str, row[:10])) for row in x.tolist()]
            inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.bert(**inputs).last_hidden_state.mean(dim=1)
            return self.fc(outputs).cpu().detach().numpy()



    bert_model = BERTFeatureExtractor().eval()
    X_train_bert = bert_model(torch.tensor(X_train, dtype=torch.float32))
    X_test_bert = bert_model(torch.tensor(X_test, dtype=torch.float32))

    st.write("🟡 **Training XGBoost Model...**")
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_clf.fit(X_train_bert, y_train)

    y_pred = xgb_clf.predict(X_test_bert)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ **Model Accuracy:** {accuracy:.4f}")

    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=["Healthy", "Schizophrenia"])

    # Display classification report in Streamlit
    st.write("📊 **Classification Report:**")
    st.text(class_report)  # Displaying the report as plain text

    # Existing confusion matrix code
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Schizophrenia"])
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)
    #bar graph
    healthy_count = np.sum(y == 0)
    schizophrenia_count = np.sum(y == 1)

    fig, ax = plt.subplots()
    bars = ax.bar(["Healthy", "Schizophrenia"], [healthy_count, schizophrenia_count], color=["blue", "red"])

    ax.set_xlabel("Condition")
    ax.set_ylabel("Count")
    ax.set_title("Dataset Distribution")

    # Add labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=12,
                fontweight='bold')

    st.pyplot(fig)

    st.sidebar.header("Upload EEG File for Prediction")
    pred_file = st.sidebar.file_uploader("Upload a single EEG file (.edf)", type=["edf"])

    if pred_file:
        temp_file = "temp.edf"
        with open(temp_file, "wb") as f:
            f.write(pred_file.getbuffer())

        pred_data = load_eeg(temp_file)
        pred_features = extract_features(pred_data)
        pred_features = scaler.transform([pred_features])
        pred_features_bert = bert_model(torch.tensor(pred_features, dtype=torch.float32))

        pred_label = xgb_clf.predict(pred_features_bert)[0]
        diagnosis = "Healthy" if pred_label == 0 else "Schizophrenia"
        st.sidebar.success(f"🧠 **Prediction:** {diagnosis}")

    # Sample EEG data (replace this with actual EEG feature values)
    np.random.seed(42)
    feature_1 = np.random.normal(loc=0.5, scale=0.1, size=100)  # Simulated EEG feature 1
    feature_2 = np.random.normal(loc=0.3, scale=0.1, size=100)  # Simulated EEG feature 2
    labels = np.random.choice([0, 1], size=100)  # 0: Healthy, 1: Schizophrenia

    # Sample EEG data (replace this with actual EEG feature values)
    np.random.seed(42)
    feature_1 = np.random.normal(loc=0.5, scale=0.1, size=100)  # Simulated EEG feature 1
    feature_2 = np.random.normal(loc=0.3, scale=0.1, size=100)  # Simulated EEG feature 2
    labels = np.random.choice([0, 1], size=100)  # 0: Healthy, 1: Schizophrenia


    def plot_scatter_with_line():
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot points with different colors for Healthy (0) and Schizophrenia (1)
        for label, color in zip([0, 1], ["blue", "red"]):
            mask = labels == label
            ax.scatter(feature_1[mask], feature_2[mask], label="Healthy" if label == 0 else "Schizophrenia",
                       color=color, alpha=0.6)

        # Add a reference line (Decision boundary or trend line)
        x_vals = np.linspace(min(feature_1), max(feature_1), 100)
        y_vals = 0.5 * x_vals + 0.1  # Example equation: y = 0.5x + 0.1
        ax.plot(x_vals, y_vals, color="black", linestyle="dashed", linewidth=2, label="Reference Line")

        ax.set_xlabel("EEG Feature 1")
        ax.set_ylabel("EEG Feature 2")
        ax.set_title("EEG Feature Scatter Plot with Reference Line")
        ax.legend()

        return fig


    st.pyplot(plot_scatter_with_line())


    if pred_file:
        temp_file = "temp.edf"
        with open(temp_file, "wb") as f:
            f.write(pred_file.getbuffer())

        pred_data = load_eeg(temp_file)
        pred_features = extract_features(pred_data)
        pred_features = scaler.transform([pred_features])
        pred_features_bert = bert_model(torch.tensor(pred_features, dtype=torch.float32))

        pred_label = xgb_clf.predict(pred_features_bert)[0]
        diagnosis = "Healthy" if pred_label == 0 else "Schizophrenia"

        st.sidebar.success(f"🧠 **Prediction:** {diagnosis}")

        # **Final Diagnosis Section (Ensures pred_label exists)**
        st.subheader("🧠 Final Diagnosis Result")
        if pred_label == 0:
            st.success("✅ The EEG pattern suggests a **Healthy** brain.")
        else:
            st.error("⚠️ The EEG pattern indicates **Schizophrenia**.")
