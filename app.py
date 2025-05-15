# -*- coding: utf-8 -*-
"""
Modern Developer Metrics & Professional Identification App
Enhanced Streamlit Design with Improved Graphs & Dual Download Options (SVG/PDF)
All text is now in English. No LLM integrations.
"""

import asyncio
import os
import random
import subprocess
import tempfile
import time
import warnings
import networkx as nx
import plotly.graph_objects as go
# Import third-party libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
# scikit-learn modules for modeling and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    log_loss, fbeta_score, brier_score_loss, average_precision_score, make_scorer
)
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional libraries
try:
    import umap
except ImportError:
    umap = None

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    AutoML = None

from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set a clean and modern Plotly theme
pio.templates.default = "plotly_white"

##############################################################################
# CUSTOM CSS & PAGE CONFIGURATION
##############################################################################
# Custom CSS to enhance the overall appearance of the app
CUSTOM_CSS = """
<style>
/* Global styling for a modern interface */
body {
    background-color: #f5f5f5;
    color: #333;
    font-family: 'Helvetica Neue', Arial, sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #2c3e50;
    margin-top: 0.75em;
    margin-bottom: 0.5em;
}
table, .metrics-table {
    margin-bottom: 1em;
    border: 1px solid #ccc;
    border-collapse: collapse;
}
table th, table td, .metrics-table th, .metrics-table td {
    border: 1px solid #ccc;
    padding: 0.5em;
}
/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #eaecef;
    padding-top: 1rem !important;
}
/* Buttons */
.stButton>button {
    background-color: #3498db;
    color: white;
    border-radius: 5px;
    font-size: 16px;
    margin-bottom: 0.5em;
    height: 2.5em;
    width: 100%;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #2980b9;
}
/* Header styling */
section[data-testid="stHeader"] {
    border-bottom: 1px solid #ccc;
    margin-bottom: 1em;
}
</style>
"""

# Set page configuration and include custom CSS
st.set_page_config(page_title="Bakalarska", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

##############################################################################
# GLOBAL SETTINGS & HYPERPARAMETERS
##############################################################################
# Path to the dataset (Update if necessary)
DATASET_PATH = r"C:\Users\nikit\PycharmProjects\Bk\DataCollecting\src\results\developer_metrics_final.parquet"

# List of numeric and categorical features for processing
NUMERIC_FEATURES = [
    "followers", "days", "weeks", "period", "timediff", "timediff_std",
    "commits", "weekend", "night", "morning", "afternoon", "evening",
    "office", "most_active_hour", "beginning_regular", "end_regular",
    "length_regular", "avg_message_len", "loc_per_commit",
    "max_loc_in_one_commit", "loc_mean_per_commit", "avg_loc_per_day",
    "total_repositories_fetched"
]

CATEGORICAL_FEATURES = ["username", "email", "skuseny", "neznamy"]

# Hyperparameters for RandomForest and LightGBM models
HYPERPARAM_CONFIG = {
    "RandomForest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": ["auto", 5, 10],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2","auto"],
        "bootstrap": [True, False]
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40],
        "subsample": [0.8, 1.0],
        "min_split_gain": [0.1, 0.2],
        "min_data_in_leaf": [20, 30],
        "min_sum_hessian_in_leaf": [1e-3, 0.01]
    }
}

##############################################################################
# GRAPH DOWNLOAD FUNCTIONS (SVG & PDF)
##############################################################################
def download_vector_figure(fig, filename="figure.svg"):
    """
    Converts a Plotly figure to SVG and provides a download button.
    """
    try:
        svg_bytes = fig.to_image(format="svg")
        st.download_button(
            label="Download as SVG",
            data=svg_bytes,
            file_name=filename,
            mime="image/svg+xml"
        )
    except (ValueError, OSError) as e:
        st.error(f"Cannot export to SVG: {e}")


def convert_svg_to_pdf(svg_bytes, pdf_filename="figure.pdf"):
    """
    Converts an SVG byte stream to PDF using Inkscape.
    Inkscape must be installed locally for this to work.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
            tmp_svg.write(svg_bytes)
            tmp_svg_path = tmp_svg.name

        tmp_pdf_path = os.path.join(os.path.dirname(tmp_svg_path), pdf_filename)
        # Path to the Inkscape executable (update if necessary)
        inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"
        if not os.path.exists(inkscape_path):
            raise FileNotFoundError(f"Inkscape not found at: {inkscape_path}")

        subprocess.run([
            inkscape_path,
            tmp_svg_path,
            "--export-filename", tmp_pdf_path,
            "--export-area-drawing",
            "--export-text-to-path"
        ], check=True)

        if not os.path.exists(tmp_pdf_path):
            raise FileNotFoundError(f"PDF was not created: {tmp_pdf_path}")

        with open(tmp_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        os.remove(tmp_svg_path)
        os.remove(tmp_pdf_path)
        return pdf_bytes

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        st.error(f"Error converting SVG->PDF: {e}")
        return None


def download_pdf_vector_figure(fig, filename="figure.pdf"):
    """
    Converts a Plotly figure to PDF and provides a download button.
    """
    try:
        svg_bytes = fig.to_image(format="svg")
        pdf_bytes = convert_svg_to_pdf(svg_bytes, filename)
        if pdf_bytes:
            st.download_button(
                label="Download as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf"
            )
    except (ValueError, OSError) as e:
        st.error(f"Cannot export to PDF: {e}")

##############################################################################
# DATA LOADING & PREPROCESSING
##############################################################################
@st.cache_data(show_spinner=True)
def load_dataset(path=DATASET_PATH):
    """
    Loads the dataset from a Parquet file.
    Ensures that categorical columns are converted to string type.
    """
    if not os.path.exists(path):
        st.error(f"Dataset not found at '{path}'. Please check the path.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        # Ensure categorical columns are strings.
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("string")
        # Let pandas infer better types for remaining columns.
        df = df.convert_dtypes()
        return df
    except (OSError, ValueError) as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


def generate_profiling_report(df: pd.DataFrame) -> None:
    """
    Vygeneruje a zobrazí HTML profilovací report pre daný DataFrame
    s využitím knižnice ydata-profiling. Report sa vloží do rozklikávacieho
    expanderu, aby bolo možné ho jednoducho skryť a pokračovať v práci
    s ostatnými časťami aplikácie. Dodatočne je dostupné tlačidlo na
    stiahnutie reportu v HTML formáte.
    """
    profile = ProfileReport(
        df,
        title="Automatizovaná analýza dát",
        minimal=True
    )

    # Генеруємо HTML-звіт
    profile_html = profile.to_html()

    st.download_button(
        label="Stiahnuť Profilovací Report (HTML)",
        data=profile_html,
        file_name="profiling_report.html",
        mime="text/html"
    )

    with st.expander("Zobraziť/Skryť Profilovací Report", expanded=False):
        components.html(
            profile_html,
            height=1200,  # За потреби змініть висоту
            scrolling=True
        )

def convert_time_to_float(time_val):
    """
    Converts a time value in 'HH:MM' format to a float representing hours.
    For example, '09:30' becomes 9.5.
    Returns np.nan if conversion fails.
    """
    try:
        if isinstance(time_val, str) and ":" in time_val:
            parts = time_val.split(":")
            hours = float(parts[0])
            minutes = float(parts[1])
            return hours + minutes / 60.0
        else:
            # Try to convert directly to float.
            return float(time_val)
    except (ValueError, TypeError):
        return np.nan

def convert_percentage(val):
    """
    Converts a percentage value (e.g. '45.5%') to a float fraction (0.455).
    If the input is already numeric, returns it as a float.
    """
    try:
        if isinstance(val, str) and "%" in val:
            val_clean = val.replace("%", "").strip()
            return float(val_clean) / 100.0
        else:
            return float(val)
    except (ValueError, TypeError):
        return np.nan

def impute_numeric_column(series):
    """
    Imputes missing values in a numeric Series.
    If the entire column is missing, fills with 0.
    Otherwise, fills missing values with the median.
    """
    if series.isna().all():
        return series.fillna(0)
    else:
        median_value = series.median()
        return series.fillna(median_value)

def convert_target_value(val):
    """
    Converts the target column value (skuseny) to an integer.
    Returns 1 if the value represents True (case-insensitive), 0 otherwise.
    Handles boolean values as well as strings.
    """
    try:
        if isinstance(val, bool):
            return int(val)
        elif isinstance(val, str):
            if val.strip().lower() == "true":
                return 1
            else:
                return 0
        else:
            return int(val)
    except (ValueError, TypeError):
        return 0


def preprocess_data(df):
    """
    Preprocesses the input DataFrame in several clear steps:

      1. Time Conversion: Converts columns like 'most_active_hour', 'beginning_regular',
         'end_regular', and 'office' from 'HH:MM' strings to float hours.

      2. Percentage Processing: Processes columns (e.g. 'weekend', 'night', 'morning', 'afternoon', 'evening')
         by removing the '%' symbol and converting them to float fractions.

      3. Numeric Conversion & Imputation: For all columns listed in NUMERIC_FEATURES,
         converts values to numeric types and imputes missing values using the median. If an entire column is missing,
         it is filled with 0.

      4. Target Creation: The target variable 'y' is created from the 'skuseny' column,
         using a dedicated conversion function.

      5. Final Sanity: As a safeguard, any remaining NaN values in the features are filled with 0.

    Returns:
      X: A DataFrame of processed numeric features with no missing values.
      y: The target series.
    """
    df_processed = df.copy()

    # 1. Time Conversion
    time_cols = ["most_active_hour", "beginning_regular", "end_regular", "office"]
    for col in time_cols:
        if col in df_processed.columns:
            new_values = []
            for x in df_processed[col]:
                new_values.append(convert_time_to_float(x))
            df_processed[col] = pd.Series(new_values, index=df_processed.index)

    # 2. Percentage Processing
    perc_cols = ["weekend", "night", "morning", "afternoon", "evening"]
    for col in perc_cols:
        if col in df_processed.columns:
            new_values = []
            for x in df_processed[col]:
                new_values.append(convert_percentage(x))
            df_processed[col] = pd.Series(new_values, index=df_processed.index)

    # 3. Numeric Conversion & Imputation
    valid_numeric_cols = [col for col in NUMERIC_FEATURES if col in df_processed.columns]
    X = df_processed[valid_numeric_cols].copy()
    for col in valid_numeric_cols:
        # Convert to numeric type (coercing errors to NaN)
        X[col] = pd.to_numeric(X[col], errors='coerce')
        # Impute missing values using our helper
        X[col] = impute_numeric_column(X[col])

    # Final safeguard: fill any remaining NaNs with 0.
    X.fillna(0, inplace=True)

    # 4. Target Creation
    if "skuseny" in df_processed.columns:
        y = df_processed["skuseny"].apply(convert_target_value)
    else:
        st.error("Target column 'skuseny' not found in the dataset.")
        y = pd.Series([0] * len(df_processed))

    return X, y

##############################################################################
# FEATURE SELECTION METHODS
##############################################################################
def univariate_feature_selection(X, y, k=10):  # Try a smaller k, e.g., 10
    """
    Selects the top k features using the ANOVA F-test.
    If there's only one class in y, a warning is issued.

    Returns:
      A tuple (X_new, features) where X_new is a DataFrame of the selected features
      and features is the list of selected feature names.
    """
    if len(np.unique(y)) < 2:
        st.warning("Target variable has only one unique class. Univariate feature selection may not be meaningful.")

    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_feats = X.columns[mask].tolist()
    return pd.DataFrame(X_new, columns=selected_feats), selected_feats


def recursive_feature_elimination(X, y, estimator=None, n_features=10):  # Try a smaller n_features
    """
    Performs Recursive Feature Elimination (RFE) using a given estimator (default: LogisticRegression).

    Returns:
      A tuple (X_selected, features) where X_selected is the DataFrame with selected features
      and features is the list of selected feature names.
    """
    if estimator is None:
        estimator = LogisticRegression(solver='liblinear', max_iter=1000)

    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
    rfe.fit(X, y)
    mask = rfe.support_
    selected_feats = X.columns[mask].tolist()
    return X[selected_feats].copy(), selected_feats


def combined_feature_selection(X, y):
    """
    Combines Univariate Feature Selection and Recursive Feature Elimination (RFE).
    The resulting feature set includes any feature selected by either method.

    Returns:
      A tuple (X_combined, combined_features) where X_combined is the DataFrame containing only the
      combined selected features and combined_features is a list of their names.
    """

    # 1. Univariate Feature Selection
    st.markdown("<h4 style='color:#2c3e50;'>1) Univariate Feature Selection</h4>", unsafe_allow_html=True)
    X_uni, feats_uni = univariate_feature_selection(X, y, k=10)  # Adjust k here as needed
    with st.expander("Univariate Selection Details"):
        st.success("Selected Features (Univariate):")
        st.write(feats_uni)

    # 2. Recursive Feature Elimination (RFE)
    st.markdown("<h4 style='color:#2c3e50;'>2) Recursive Feature Elimination (RFE)</h4>", unsafe_allow_html=True)
    X_rfe, feats_rfe = recursive_feature_elimination(X, y, n_features=10)  # Adjust n_features here as needed
    with st.expander("RFE Details"):
        st.info("Selected Features (RFE):")
        st.write(feats_rfe)

    # Combine features from both methods
    combined_features = sorted(set(feats_uni + feats_rfe))

    st.markdown("<h4 style='color:#2c3e50;margin-top:30px;'>Combined Feature Set</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:15px;'>These features were selected by either Univariate or RFE methods.</p>",
                unsafe_allow_html=True)
    with st.expander("See Combined Feature List"):
        st.markdown("<ul>", unsafe_allow_html=True)
        for feat in combined_features:
            st.markdown(f"<li style='font-size:14px;'>{feat}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("<h5 style='color:#2c3e50;margin-top:20px;'>Final Selected Features</h5>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;'>Below is a table listing the combined features.</p>",
                unsafe_allow_html=True)
    feature_df = pd.DataFrame({"Feature": combined_features})
    feature_df.index += 1
    st.table(feature_df)

    return X[combined_features].copy(), combined_features

##############################################################################
# EXPERIMENTAL DEEP LEARNING MODEL (PyTorch)
##############################################################################
class ExperimentalNet(nn.Module):
    """
    A simple feed-forward neural network with dropout and batch normalization.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


def experimental_deep_learning_model(X, y, epochs=30, batch_size=32, validation_split=0.2):
    if len(X) < 2 or X.shape[1] == 0:
        raise ValueError("Not enough data/features to train the deep learning model.")

    device = torch.device("cpu")  # Use 'cuda' if available

    X_array = X.values.astype(np.float32)
    y_array = y.values.astype(np.float32).reshape(-1, 1)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
    )

    tv_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_trainval).to(device),
        torch.tensor(y_trainval).to(device)
    )
    val_size = int(validation_split * len(tv_dataset))
    train_size = len(tv_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        tv_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test).to(device),
        torch.tensor(y_test).to(device)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ExperimentalNet(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        epoch_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item() * xb.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        progress_bar.progress(int((epoch + 1) / epochs * 100))
    total_time = time.time() - start_time

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            output = model(xb)
            preds = (output >= 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    all_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            prob_output = model(xb)
            all_probs.append(prob_output.cpu().numpy())
    all_probs = np.concatenate(all_probs).flatten()

    metrics = {
        "Accuracy": accuracy_score(all_targets, all_preds),
        "Precision": precision_score(all_targets, all_preds, zero_division=0),
        "Recall": recall_score(all_targets, all_preds, zero_division=0),
        "F1 Score": f1_score(all_targets, all_preds, zero_division=0),
        "ROC AUC": roc_auc_score(all_targets, all_probs),
        "Training Time (s)": total_time
    }
    return model, metrics, history

##############################################################################
# TRAINING & EVALUATION FOR CLASSIFICATION MODELS
##############################################################################
def train_classification_model(X, y, model_type="RandomForest", test_size=0.25):
    """
    Trains a classification model using a pipeline and randomized search with multiple scoring metrics.
    The data is split into training and hold‐out test sets. Hyperparameter tuning is performed on the
    training set using RandomizedSearchCV with StratifiedKFold, and the tuned model is evaluated on the test set.

    Parameters:
      - X: Feature matrix.
      - y: Target vector.
      - model_type: "RandomForest" or "LightGBM".
      - test_size: Fraction of data to reserve for testing.

    Returns:
      - best_model: The best estimator from hyperparameter tuning.
      - best_params: Best hyperparameters.
      - test_metrics: Performance metrics on the hold‐out test set.
    """
    # Split data into training and hold-out test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Select classifier and hyperparameter grid
    if model_type == "RandomForest":
        clf = RandomForestClassifier(random_state=42)
        param_dist = HYPERPARAM_CONFIG["RandomForest"]
    else:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
        param_dist = HYPERPARAM_CONFIG["LightGBM"]

    # Build pipeline with scaling and classifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

    # Define scoring metrics
    scoring = {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1': 'f1',
        'Balanced Accuracy': 'balanced_accuracy',
        "Cohen's Kappa": make_scorer(cohen_kappa_score),
        'Matthews Corrcoef': make_scorer(matthews_corrcoef),
        'ROC AUC': 'roc_auc',
        'Log Loss': 'neg_log_loss',
        'F2 Score': make_scorer(fbeta_score, beta=2, zero_division=0),
        'Brier Score': make_scorer(brier_score_loss, greater_is_better=False),
        'Avg Precision': 'average_precision'
    }

    # Perform hyperparameter tuning on the training set
    search = RandomizedSearchCV(
        pipe,
        param_distributions={f"classifier__{k}": v for k, v in param_dist.items()},
        n_iter=20,
        cv=3,
        scoring=scoring,
        refit='F1',
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    # Additional cross-validation on training data (computed but not displayed here)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _ = cross_validate(best_model, X_train, y_train, cv=skf, scoring=scoring, return_train_score=True)

    # Evaluate on the hold-out test set
    test_metrics, _ = evaluate_model_performance(best_model, X_test, y_test)
    return best_model, best_params, test_metrics


def evaluate_model_performance(model, X, y):
    """
    Evaluates the trained model on the provided dataset.
    Computes performance metrics and returns them for display.

    Parameters:
      - model: Trained classifier.
      - X: Feature matrix.
      - y: True labels.

    Returns:
      - metrics_dict: A dictionary of computed performance metrics.
      - y_pred: Predicted labels.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    metrics_dict = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y, y_pred),
        "Matthews Corrcoef": matthews_corrcoef(y, y_pred)
    }
    if y_proba is not None:
        try:
            metrics_dict["ROC AUC"] = roc_auc_score(y, y_proba[:, 1])
            metrics_dict["Log Loss"] = log_loss(y, y_proba)
            metrics_dict["F2 Score"] = fbeta_score(y, y_pred, beta=2, zero_division=0)
            metrics_dict["Brier Score"] = brier_score_loss(y, y_proba[:, 1])
            metrics_dict["Avg Precision"] = average_precision_score(y, y_proba[:, 1])
        except (ValueError, IndexError) as e:
            pass
    return metrics_dict, y_pred

##############################################################################
# ADVANCED VISUALIZATION & INTERPRETATION TOOLS (UMAP,FLAML)
##############################################################################
def visualize_umap(X, y):
    """
    Reduces dimensions via UMAP and plots a 2D scatter with a boolean 'professional' label.
    Provides download options for the plot.
    """
    if umap is None:
        st.error("UMAP is not installed. Please install 'umap-learn' first.")
        return

    st.subheader("UMAP Configuration")
    n_neighbors = st.slider("Number of Neighbors", 5, 50, 15)
    min_dist = st.slider("Min Distance", 0.0, 1.0, 0.1, step=0.05)

    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    emb = reducer.fit_transform(X)

    # Build dataframe with boolean professional flag
    df_emb = pd.DataFrame({
        "Dimenzia 1": emb[:, 0],
        "Dimenzia 2": emb[:, 1],
        "professional": y.astype(bool)
    })

    # Plot
    fig = px.scatter(
        df_emb,
        x="Dimenzia 1",
        y="Dimenzia 2",
        color="professional",
        title="UMAP 2D projekcia (Profesionál = True / False)",
        labels={"professional": "Profesionál"}
    )

    # Localize axis and legend titles
    fig.update_layout(
        xaxis_title="Dimenzia 1",
        yaxis_title="Dimenzia 2",
        legend_title_text="Profesionál?"
    )

    st.plotly_chart(fig, use_container_width=True)
    download_vector_figure(fig, "umap_embedding.svg")
    download_pdf_vector_figure(fig, "umap_embedding.pdf")


@st.cache_resource
def train_flaml_model(X, y, time_budget=60):
    """
    Trains an AutoML model using FLAML within the specified time budget.
    Returns the trained AutoML model.
    """
    if not FLAML_AVAILABLE:
        st.error("FLAML is not installed.")
        return None

    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": "f1",
        "task": "classification",
        "log_file_name": "flaml_developer.log",
        "seed": 42
    }
    automl.fit(X, y, **settings)
    return automl

##############################################################################
# DATA OVERVIEW FUNCTIONS
##############################################################################
def dataset_summary_info(df: pd.DataFrame):
    """
    Generates a summary of the DataFrame including non-null counts, data types,
    unique values, and total memory usage.
    Returns:
      - summary DataFrame
      - memory usage in KB
    """
    data = []
    total_rows = len(df)
    for col in df.columns:
        non_null_count = df[col].notnull().sum()
        null_count = total_rows - non_null_count
        dtype = df[col].dtype
        uniq_vals = df[col].nunique(dropna=True) if df[col].notna().any() else 0

        example_val = None
        non_null_sample = df[col].dropna()
        if len(non_null_sample) > 0:
            example_val = non_null_sample.iloc[0]

        data.append([
            col,
            non_null_count,
            null_count,
            str(dtype),
            uniq_vals,
            str(example_val)[:50]
        ])

    info_df = pd.DataFrame(data, columns=[
        "Column", "Non-Null Count", "Null Count", "Dtype", "Unique Values", "Example Value"
    ])
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_kb = mem_bytes / 1024
    return info_df, mem_kb

##############################################################################
# STREAMLIT PAGE DEFINITIONS
##############################################################################
def page_data_and_model_training():
    """
    Page for:
      - Dataset overview
      - Feature selection
      - Classification model training & evaluation
    """
    # Custom CSS for modern styling
    st.markdown("""
        <style>
            .page-title {
                text-align: center;
                font-size: 2.8rem;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 0.5rem;
            }
            .section-header {
                text-align: center;
                font-size: 2rem;
                font-weight: 600;
                color: #34495e;
                margin-bottom: 1rem;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.3rem;
            }
            .subheader {
                font-size: 1.5rem;
                color: #2c3e50;
            }
            .custom-metric {
                font-size: 1.2rem;
                color: #2980b9;
                font-weight: 500;
            }
            hr {
                border: none;
                border-top: 2px solid #ecf0f1;
                margin: 2rem 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='page-title'>Data & Model Training</div>", unsafe_allow_html=True)

    # Dataset Overview Section
    with st.container():
        st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
        df = load_dataset()
        if df.empty:
            st.warning("No data loaded, cannot proceed.")
            return
        if st.button("Generate Data Profiling Report"):
            with st.spinner("Generating profiling report..."):
                generate_profiling_report(df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.subheader("Dataset Summary")
            summary_df, mem_kb = dataset_summary_info(df)
            st.dataframe(summary_df, use_container_width=True)
            st.markdown(f"<h4>Total Memory Usage: {mem_kb:.2f} KB</h4>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Feature Selection Section
    with st.container():
        st.markdown("<div class='section-header'>Feature Selection</div>", unsafe_allow_html=True)
        X, y = preprocess_data(df)
        if X.empty:
            st.error("No numeric features found. Please check the dataset.")
            return

        X_sel, selected_feats = combined_feature_selection(X, y)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Model Training & Evaluation Section
    with st.container():
        st.markdown("<div class='section-header'>Classification Model Training</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:1.1rem; color:#7f8c8d;'>Select the model type and start training the classifier.</p>", unsafe_allow_html=True)
        chosen_model_type = st.selectbox("Choose Model Type", ["RandomForest", "LightGBM"])

        if st.button("Train Classifier"):
            with st.spinner("Training the classifier..."):
                best_model, best_params, test_metrics = train_classification_model(X_sel, y, chosen_model_type)


            # Display best hyperparameters
            st.markdown("<h4>Best Hyperparameters</h4>", unsafe_allow_html=True)
            st.table(pd.DataFrame([best_params]))

            # Display test set evaluation metrics
            st.markdown("<h4>Test Set Evaluation Metrics</h4>", unsafe_allow_html=True)
            st.table(pd.DataFrame(test_metrics.items(), columns=["Metric", "Value"]))

            st.success("Model training completed!")
            # Save results to session state for later use
            st.session_state["best_model"] = best_model
            st.session_state["X_selected"] = X_sel
            st.session_state["y"] = y



def page_advanced_tools_and_interpretations():
    """
    Page for advanced visualization & model interpretation:
      - UMAP for dimensionality reduction
      - FLAML for AutoML
    """
    st.title("Advanced Tools & Interpretations")

    if "best_model" not in st.session_state:
        st.warning("Please train a model first on the 'Data & Model Training' page.")
        return

    st.header("UMAP Visualization")
    df = load_dataset()
    if df.empty:
        st.warning("No data available for UMAP visualization.")
        return
    X, y = preprocess_data(df)

    if "show_umap" not in st.session_state:
        st.session_state["show_umap"] = False

    # --- Funkcia na prepnutie stavu ---
    def _toggle_umap():
        st.session_state["show_umap"] = not st.session_state["show_umap"]

    # --- Dynamický label tlačidla podľa stavu ---
    btn_label = "Skryť UMAP graf" if st.session_state["show_umap"] else "Zobraziť UMAP graf"
    st.button(btn_label, on_click=_toggle_umap)

    # --- Kondicionálne vykreslenie alebo skrytie grafu ---
    if st.session_state["show_umap"]:
        visualize_umap(X, y)

    st.header("FLAML AutoML")
    st.write("Run an automated model search within a set time budget.")
    time_budget = st.slider("Time Budget (in seconds)", 10, 300, 60, step=10)
    if st.button("Run FLAML AutoML"):
        if len(X) < 2:
            st.error("Not enough data to run FLAML.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        automl = train_flaml_model(X_train, y_train, time_budget)
        if automl:
            st.write("**Best Config:**", automl.best_config)
            preds = automl.predict(X_test)
            acc_ = accuracy_score(y_test, preds)
            st.write(f"**Test Accuracy:** {acc_:.4f}")

def render_model_architecture(input_dim: int):
    """
    Vykreslí architektúru ExperimentalNet pomocou NetworkX + Plotly.
    Umožňuje interaktívne zobrazenie a stiahnutie SVG/PDF.
    """
    # Definícia vrstiev a ich uzlov (len textové popisky sú po slovensky)
    layers = [
        [('Vstup', f'{input_dim} príznakov')],
        [('FC1', '64 neurónov'), ('ReLU1', ''), ('BN1', '64'), ('Dropout1', 'p=0,3')],
        [('FC2', '128 neurónov'), ('ReLU2', ''), ('BN2', '128'), ('Dropout2', 'p=0,3')],
        [('FC3', '64 neurónov'), ('ReLU3', ''), ('BN3', '64'), ('Dropout3', 'p=0,3')],
        [('Výstup', '1 neurón')]
    ]

    # Zostavíme NetworkX graf
    G = nx.DiGraph()
    pos = {}
    x_gap = 300
    y_gap = 100
    for i, layer in enumerate(layers):
        n = len(layer)
        for j, (name, label) in enumerate(layer):
            G.add_node(name, label=label)
            pos[name] = (i * x_gap, (j - (n - 1) / 2) * y_gap)

    # Spojenia (vstup ↗ skryté vrstvy ↗ výstup)
    edges = [
        ('Vstup','FC1'),
        ('FC1','ReLU1'), ('ReLU1','BN1'), ('BN1','Dropout1'),
        ('Dropout1','FC2'),
        ('FC2','ReLU2'), ('ReLU2','BN2'), ('BN2','Dropout2'),
        ('Dropout2','FC3'),
        ('FC3','ReLU3'), ('ReLU3','BN3'), ('BN3','Dropout3'),
        ('Dropout3','Výstup')
    ]
    G.add_edges_from(edges)

    # Vytvorenie Plotly figúry
    fig = go.Figure()

    # Kreslíme hrany
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=2),
            hoverinfo='none'
        ))

    # Kreslíme uzly s popiskami
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        text = node if not data['label'] else f"{node}\n{data['label']}"
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=60, line=dict(width=2)),
            text=[text],
            textposition='bottom center',
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Architektúra experimentálnej siete",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    # Zobrazenie a stiahnutie
    st.plotly_chart(fig, use_container_width=True)
    svg = fig.to_image(format="svg")
    st.download_button("Stiahnuť architektúru (SVG)", svg, "architecture.svg", "image/svg+xml")
    try:
        pdf = fig.to_image(format="pdf")
        st.download_button("Stiahnuť architektúru (PDF)", pdf, "architecture.pdf", "application/pdf")
    except Exception:
        st.warning("PDF export nie je k dispozícii.")




def page_experimental_dl():
    st.title("Experimental Deep Learning")

    df = load_dataset()
    if df.empty:
        st.warning("Dataset not found. Cannot proceed with DL training.")
        return

    X, y = preprocess_data(df)
    if X.empty:
        st.error("No numeric features available. Cannot train the deep learning model.")
        return

    st.write("Tune hyperparameters for the deep learning model:")
    epochs = st.slider("Epochs", 1, 100, 30)
    batch_size = st.slider("Batch Size", 8, 256, 32, step=8)
    validation_split = st.slider("Validation Split", 0.05, 0.5, 0.2, step=0.05)

    if "dl_trained" not in st.session_state:
        st.session_state["dl_trained"] = False

    def _run_training():
        model, metrics_dict, history = experimental_deep_learning_model(
            X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split
        )
        st.session_state["dl_metrics"] = metrics_dict
        st.session_state["dl_history"] = history
        st.session_state["dl_input_dim"] = X.shape[1]
        st.session_state["dl_trained"] = True

    btn_label = "Skryť výsledky" if st.session_state["dl_trained"] else "Spustiť tréning DL"
    if st.button(btn_label, on_click=_run_training if not st.session_state["dl_trained"] else lambda: st.session_state.update({"dl_trained": False})):
        pass

    if st.session_state["dl_trained"]:
        st.subheader("Final Test Metrics")
        st.table(pd.DataFrame(
            st.session_state["dl_metrics"].items(),
            columns=["Metric", "Value"]
        ))

        render_model_architecture(input_dim=st.session_state["dl_input_dim"])

        st.subheader("Training Curves")
        hist = st.session_state["dl_history"]
        hist_df = pd.DataFrame({
            "Tréningová strata": hist["train_loss"],
            "Validačná strata": hist["val_loss"]
        })
        fig_curve = px.line(
            hist_df,
            labels={
                "index": "Epocha",
                "value": "Hodnota straty",
                "variable": "Typ straty"
            },
            title="Priebeh tréningovej a validačnej straty počas epoch"
        )
        fig_curve.update_layout(
            xaxis_title="Epocha",
            yaxis_title="Strata",
            legend_title_text="Legenda"
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        download_vector_figure(fig_curve, "training_loss.svg")
        download_pdf_vector_figure(fig_curve, "training_loss.pdf")

##############################################################################
# MAIN FUNCTION
##############################################################################
def main():
    """
    Main function to control page navigation using the Streamlit sidebar.
    """
    st.sidebar.title("Navigation")
    options = [
        "Data & Model Training",
        "Advanced Tools & Interpretations",
        "Experimental Deep Learning"
    ]
    choice = st.sidebar.radio("Go to section:", options)

    if choice == "Data & Model Training":
        page_data_and_model_training()
    elif choice == "Advanced Tools & Interpretations":
        page_advanced_tools_and_interpretations()
    elif choice == "Experimental Deep Learning":
        page_experimental_dl()


if __name__ == "__main__":
    main()
