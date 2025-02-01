# -*- coding: utf-8 -*-
"""
Autor: <Vaše meno>
Dátum: 2025
Popis:
Rozšírená demonštračná aplikácia v Streamlit, ktorá ilustruje:
 - FLAML (AutoML)
 - Viac algoritmov klastrovania (KMeans, DBSCAN, AgglomerativeClustering) s automatickým výberom počtu klastrov
 - Pipeline (Feature weighting + StandardScaler + klaster)
 - RandomizedSearchCV s K-Fold cross-validation pre optimalizáciu RandomForest s rozšírenými hyperparametrami
 - Rozšírené metriky: Accuracy, Precision, Recall, F1, Balanced Accuracy, Cohen's Kappa,
   Matthews Corrcoef, ROC AUC, Log Loss
 - Interpretáciu modelu pomocou LIME (textový a interaktívny grafický výstup)
 - Interaktívne grafy s Plotly (PCA scatter plot, Confusion Matrix, barplot rozdelenia rolí, rating)
 - Kešovanie dát a výsledkov AutoML pre zrýchlenie
 - Stránkovú navigáciu a sidebar s inštrukciami

Poznámka:
- Potrebné knižnice: lime, flaml[automl], seaborn, plotly, scikit-learn, atď.
- Spustenie: streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

# Pre interaktívne grafy s Plotly
import plotly.express as px
import plotly.io as pio
import plotly.tools as tls  # pre konverziu matplotlib -> Plotly

pio.templates.default = "plotly_white"

import seaborn as sns
import matplotlib.pyplot as plt

# LIME pre interpretáciu modelov
from lime.lime_tabular import LimeTabularExplainer

# ML knižnice
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# FLAML (AutoML)
try:
    from flaml import AutoML

    FLAML_AVAILABLE = True
except ImportError:
    AutoML = None
    FLAML_AVAILABLE = False

#############################################
# DATA FILE – Upravte cestu podľa potreby
#############################################
DATA_FILE = "C:/Users/nikit/PycharmProjects/Bk/Data/dataset_developers_metrics.csv"

#############################################
# KONFIGURAČNÉ PARAMETRE (lze upravovať v samostatnom súbore)
#############################################
# Konfigurácia pre klastrovanie – nastavenia jednotlivých algoritmov
clustering_config = {
    "KMeans": {"n_clusters": 3, "n_init": 10},  # Predvolený počet klastrov (bude možné automaticky vybrať)
    "AgglomerativeClustering": {"n_clusters": 3, "linkage": "ward"},
    "DBSCAN": {"eps": 0.5, "min_samples": 5}
}

# Konfigurácia pre RandomForest – rozšírené parametre vrátane class_weight pre riešenie disbalansu tried
classification_config = {
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "class_weight": ["balanced", None]
    }
}

# Dynamický job mapping – mapuje hodnoty z 'job' na kategórie podľa skúseností, rolí, atď.
job_mapping = {
    "SE": "prof",
    "SSE": "skuseny",
    "SA": "skuseny",
    # Pridajte ďalšie mapovania podľa potreby...
}

# Konfigurácia pre váhovanie príznakov – nastavte koeficienty pre dôležitejšie metriky
feature_weights = {
    "CII": 1.5,
    "CNII": 1.5,
    "AddLOC": 1.2,
    "DelLOC": 1.2,
    # Pridajte ďalšie váhy podľa dôležitosti metrik
}


#############################################
# CUSTOM TRANSFORMER PRE FEATURE WEIGHTING
#############################################
class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Transformer, ktorý násobí vybrané príznaky danými váhami.
    """

    def __init__(self, weights=None):
        self.weights = weights if weights is not None else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, weight in self.weights.items():
            if col in X.columns:
                X[col] = X[col] * weight
        return X


#############################################
# FUNKCIE PRE NAČÍTANIE A PRÍPRAVU DÁT
#############################################
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    """
    Načíta dataset zo zadaného súboru a vráti pandas DataFrame.
    Pre konverziu stĺpcov s objektmi sa aplikuje astype(str) pre kompatibilitu s Arrow.
    """
    df = pd.read_csv(path)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=True)
def prepare_data(df: pd.DataFrame):
    """
    Pripraví dáta pre analýzu:
      - Pre numerické metriky: zahrnie len metriky definované v metrics_list, ak sú k dispozícii,
        inak použije všetky numerické stĺpce. Imputácia chýbajúcich hodnôt mediánom.
      - Pre kategóriové stĺpce (napr. 'name' a 'project'): vykoná Label Encoding a pripojí
        projektové metriky (frekvencia výskytu projektu).
      - Dynamická mapácia cieľovej premennej 'job' pomocou job_mapping.

    Vracia:
      - df_features: finálny DataFrame obsahujúci numerické aj zakódované kategórie.
      - y_encoded: zakódované triedy (cieľová premenná) pre klasifikáciu.
      - le: inštancia LabelEncoder pre 'job_class'.
    """
    metrics_list = ["followers", "NoC", "AB", "NAB", "CII", "CNII", "CE", "NCE",
                    "INEI", "IEI", "AddLGM", "DelLGM", "ChurnLGM", "NoMGM",
                    "AddLOC", "DelLOC", "churnLOC", "AddF", "DelF",
                    "AddSAM", "DelSAM", "ChurnSAM", "DiP", "ICT"]
    available_metrics = [col for col in metrics_list if col in df.columns]
    if available_metrics:
        df_metrics = df[available_metrics].copy()
    else:
        df_metrics = df.select_dtypes(include=[np.number]).copy()
    df_metrics = df_metrics.fillna(df_metrics.median())

    df_cat = pd.DataFrame(index=df.index)
    if "name" in df.columns:
        le_name = LabelEncoder()
        df_cat["name_encoded"] = le_name.fit_transform(df["name"].astype(str))
    if "project" in df.columns:
        le_project = LabelEncoder()
        df_cat["project_encoded"] = le_project.fit_transform(df["project"].astype(str))
        project_freq = df["project"].value_counts()
        df_cat["project_freq"] = df["project"].map(project_freq)

    df_features = pd.concat([df_metrics, df_cat], axis=1)

    if "job" in df.columns:
        def dynamic_job_mapping(x):
            return job_mapping.get(str(x).upper(), "neznamy")

        df["job_class"] = df["job"].apply(dynamic_job_mapping)
    else:
        st.error("Chýba stĺpec 'job'.")
        return None, None, None

    le = LabelEncoder()
    y_encoded = le.fit_transform(df["job_class"])
    return df_features, y_encoded, le


#############################################
# STRÁNKY A FUNKCIE PRE ANALÝZU A VIZUALIZÁCIU
#############################################

def page_dataset_overview():
    """
    Stránka s celkovým prehľadom datasetu – zobrazuje ukážku dát, rozšírené štatistiky
    pre číselné stĺpce, informácie o stĺpcoch a barplot rozdelenia rolí.
    """
    st.header("Celkový prehľad datasetu")
    df = load_data(DATA_FILE)

    tab1, tab2, tab3 = st.tabs(["Ukážka dát", "Rozšírené štatistiky", "Informácie o stĺpcoch"])
    with tab1:
        st.subheader("Ukážka dát")
        num_rows = st.number_input("Počet riadkov na zobrazenie", min_value=5, max_value=50, value=10)
        st.dataframe(df.head(num_rows))
    with tab2:
        st.subheader("Rozšírené štatistiky pre číselné stĺpce")
        numeric_df = df.select_dtypes(include=[np.number])
        desc = numeric_df.describe().T
        desc["median"] = numeric_df.median()
        desc["missing_count"] = df[numeric_df.columns].isna().sum()
        desc["missing_percent"] = (desc["missing_count"] / len(df)) * 100
        st.dataframe(desc)
        if st.checkbox("Zobraziť korelačnú maticu"):
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                 title="Korelačná matica")
            st.plotly_chart(fig_corr, use_container_width=True)
    with tab3:
        st.subheader("Informácie o stĺpcoch")
        col_info = pd.DataFrame({
            "Typ": df.dtypes,
            "Chýbajúce hodnoty": df.isna().sum(),
            "Unikátne hodnoty": df.nunique()
        })
        st.table(col_info.astype(str))

    if "job_class" in df.columns:
        st.subheader("Rozdelenie rolí")
        job_counts = df["job_class"].value_counts().reset_index()
        job_counts.columns = ["job_class", "count"]
        fig_bar = px.bar(job_counts, x="job_class", y="count",
                         title="Rozdelenie vývojárov podľa úrovne",
                         color="job_class",
                         text="count")
        fig_bar.update_layout(transition_duration=500)
        st.plotly_chart(fig_bar, use_container_width=True)


def page_data_and_clustering():
    """
    Stránka pre klastrovanie dát pomocou rôznych algoritmov s interaktívnymi grafmi.
    Automatický výber počtu klastrov pri použití KMeans.
    """
    st.header("Klastrovanie")
    df = load_data(DATA_FILE)
    df_features, y_enc, le = prepare_data(df)
    if df_features is None:
        return

    cluster_algo = st.selectbox("Vyberte algoritmus klastrovania:",
                                ["KMeans", "DBSCAN", "AgglomerativeClustering"])

    if cluster_algo == "KMeans":
        auto_select = st.checkbox("Automatický výber počtu klastrov", value=True)
        if auto_select:
            best_silhouette = -1
            best_n_clusters = 2
            for n in range(2, 11):
                model = KMeans(n_clusters=n, random_state=42, n_init=clustering_config["KMeans"]["n_init"])
                labels = model.fit_predict(df_features)
                sil = silhouette_score(df_features, labels)
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_n_clusters = n
            st.write(f"Automaticky vybraný počet klastrov: {best_n_clusters} (Silhouette: {best_silhouette:.4f})")
            cluster_model = KMeans(n_clusters=best_n_clusters, random_state=42,
                                   n_init=clustering_config["KMeans"]["n_init"])
        else:
            params = clustering_config["KMeans"]
            n_clusters_val = st.slider("Počet klastrov", 2, 10, params["n_clusters"], 1)
            cluster_model = KMeans(n_clusters=n_clusters_val, random_state=42, n_init=params["n_init"])
    elif cluster_algo == "DBSCAN":
        params = clustering_config["DBSCAN"]
        eps_val = st.slider("eps (DBSCAN)", 0.1, 5.0, params["eps"], 0.1)
        min_samples_val = st.slider("min_samples (DBSCAN)", 1, 20, params["min_samples"], 1)
        cluster_model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    else:
        params = clustering_config["AgglomerativeClustering"]
        n_clusters_val = st.slider("Počet klastrov (Agglomerative)", 2, 10, params["n_clusters"], 1)
        linkage_val = st.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters_val, linkage=linkage_val)

    pipeline_cluster = Pipeline([
        ("scaler", StandardScaler()),
        ("cluster", cluster_model)
    ])
    pipeline_cluster.fit(df_features)
    cluster_labels = pipeline_cluster.named_steps["cluster"].labels_

    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    st.subheader("Veľkosť klastrov")
    st.table(pd.DataFrame({
        "Cluster": list(cluster_sizes.keys()),
        "Veľkosť": list(cluster_sizes.values())
    }))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(df_features)

    if len(np.unique(cluster_labels)) < 2:
        st.warning("Nedostatočný počet klastrov pre výpočet metrik.")
        sil = calinski = davies = None
    else:
        if np.min(cluster_labels) < 0:
            st.write("Poznámka: DBSCAN môže vrátiť -1 pre outliers.")
            valid_mask = (cluster_labels != -1)
            if valid_mask.sum() > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
                sil = silhouette_score(df_features[valid_mask], cluster_labels[valid_mask])
                calinski = calinski_harabasz_score(df_features[valid_mask], cluster_labels[valid_mask])
                davies = davies_bouldin_score(df_features[valid_mask], cluster_labels[valid_mask])
            else:
                sil = calinski = davies = None
        else:
            sil = silhouette_score(df_features, cluster_labels)
            calinski = calinski_harabasz_score(df_features, cluster_labels)
            davies = davies_bouldin_score(df_features, cluster_labels)

    col1, col2, col3 = st.columns(3)
    if sil is not None:
        col1.metric("Silhouette Score", f"{sil:.4f}")
    if calinski is not None:
        col2.metric("Calinski-Harabasz", f"{calinski:.2f}")
    if davies is not None:
        col3.metric("Davies-Bouldin", f"{davies:.2f}")

    if cluster_algo == "KMeans":
        inertia = cluster_model.inertia_
        st.metric("Inercia (KMeans)", f"{inertia:.2f}")

    st.markdown("### Interaktívny PCA Scatter Plot")
    fig_data = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": cluster_labels.astype(str)
    })
    fig_plotly = px.scatter(
        fig_data, x="PC1", y="PC2", color="Cluster",
        title=f"{cluster_algo} + PCA (2D)",
        hover_data=["PC1", "PC2"]
    )
    fig_plotly.update_layout(transition_duration=500)
    st.plotly_chart(fig_plotly, use_container_width=True)


def page_classification():
    """
    Stránka pre klasifikáciu s hyperparametrickým ladením a 5-Fold cross-validation.
    Používa sa rozšírený pipeline s váhovaním príznakov.
    Po vytrénovaní modelu sa vypočíta aj rating (score) pre každého vývojára.
    """
    st.header("Klasifikácia + Hyperparametrické ladenie + CrossVal")
    df = load_data(DATA_FILE)
    df_features, y_encoded, le = prepare_data(df)
    if df_features is None:
        return

    # Vloženie váhovacieho transforméra do pipeline
    pipeline_steps = [
        ("feature_weighting", FeatureWeighter(weights=feature_weights)),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ]
    pipe = Pipeline(steps=pipeline_steps)

    param_dist = {
        "rf__n_estimators": classification_config["RandomForest"]["n_estimators"],
        "rf__max_depth": classification_config["RandomForest"]["max_depth"],
        "rf__min_samples_split": classification_config["RandomForest"]["min_samples_split"],
        "rf__max_features": classification_config["RandomForest"]["max_features"],
        "rf__bootstrap": classification_config["RandomForest"]["bootstrap"],
        "rf__class_weight": classification_config["RandomForest"]["class_weight"]
    }

    st.subheader("Hľadanie najlepších parametrov")
    start_time = time.time()
    search = RandomizedSearchCV(pipe, param_dist, n_iter=5, cv=3, scoring="f1_macro", random_state=42)
    search.fit(df_features, y_encoded)
    training_time = time.time() - start_time
    st.write("Najlepšie parametre:", search.best_params_)
    st.write(f"Čas ladenia: {training_time:.2f} sekúnd")

    best_model = search.best_estimator_
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, df_features, y_encoded, cv=kf, scoring="f1_macro")
    st.write("5-Fold CrossVal F1 (macro):", cv_scores)
    st.write("Priemer F1:", np.mean(cv_scores), "±", np.std(cv_scores))

    y_pred = best_model.predict(df_features)
    acc = accuracy_score(y_encoded, y_pred)
    prec = precision_score(y_encoded, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_encoded, y_pred, average="macro", zero_division=0)
    f1_ = f1_score(y_encoded, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_encoded, y_pred)
    kappa = cohen_kappa_score(y_encoded, y_pred)
    mcc = matthews_corrcoef(y_encoded, y_pred)

    additional_metrics = {}
    if hasattr(best_model, "predict_proba"):
        try:
            y_proba = best_model.predict_proba(df_features)
            roc_auc = roc_auc_score(y_encoded, y_proba, multi_class="ovr", average="macro")
            additional_metrics["ROC AUC (macro)"] = roc_auc
            ll = log_loss(y_encoded, y_proba)
            additional_metrics["Log Loss"] = ll
        except Exception as e:
            st.warning("Nepodarilo sa vypočítať ROC AUC/Log Loss: " + str(e))

    st.markdown("**Základné metriky**:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision (macro)", f"{prec:.3f}")
    col3.metric("Recall (macro)", f"{rec:.3f}")
    col4.metric("F1 (macro)", f"{f1_:.3f}")

    st.markdown("**Rozšírené metriky**:")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Balanced Accuracy", f"{bal_acc:.3f}")
    col6.metric("Cohen's Kappa", f"{kappa:.3f}")
    col7.metric("Matthews Corrcoef", f"{mcc:.3f}")
    if additional_metrics:
        for metric_name, value in additional_metrics.items():
            st.metric(metric_name, f"{value:.3f}")

    # Výpočet ratingu – pravdepodobnosť pre kategóriu "prof" prepočítaná na skóre 0-100
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(df_features)
        prof_index = np.where(le.classes_ == "prof")[0]
        if len(prof_index) > 0:
            prof_index = prof_index[0]
            ratings = y_proba[:, prof_index] * 100
            rating_df = pd.DataFrame({"Developer": np.arange(len(ratings)), "Rating": ratings})
            rating_df = rating_df.sort_values("Rating", ascending=False)
            st.subheader("Rating vývojárov (0-100)")
            fig_rating = px.bar(rating_df, x="Developer", y="Rating",
                                title="Rating vývojárov podľa pravdepodobnosti pre 'prof'",
                                text="Rating")
            fig_rating.update_layout(transition_duration=500)
            st.plotly_chart(fig_rating, use_container_width=True)

    cm = confusion_matrix(y_encoded, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                       labels={"x": "Predikovaná trieda", "y": "Skutočná trieda"},
                       x=le.classes_, y=le.classes_,
                       title="Confusion Matrix")
    fig_cm.update_layout(transition_duration=500)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("**Classification Report:**")
    cr = classification_report(y_encoded, y_pred, target_names=le.classes_, zero_division=0)
    st.text(cr)


def page_lime_interpretation():
    """
    Stránka pre interpretáciu predikcií pomocou LIME s interaktívnou vizualizáciou.
    Vylepšený layout: možnosť voľby počtu vysvetľovaných vlastností, formátovaný textový výstup
    a pokročilý interaktívny graf s vlastným stylingom.
    """
    st.header("Interpretácia predikcií s LIME")

    # Načítanie a príprava dát
    df = load_data(DATA_FILE)
    df_features, y_encoded, le = prepare_data(df)
    if df_features is None:
        return

    # Rozdelenie dát na trénovaciu a testovaciu množinu
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # Vytvorenie pipeline s RandomForest modelom pre LIME interpretáciu
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    # Pri použití LIME je dôležité zachovať DataFrame s názvami stĺpcov.
    # Preto namiesto použitia .values použijeme pôvodný DataFrame.
    X_test_df = X_test.copy()

    st.write("Vyberte index vzorky pre vysvetlenie (0 - {}):".format(len(X_test_df) - 1))
    selected_idx = st.slider("Index vzorky", 0, len(X_test_df) - 1, 0)

    num_features = st.selectbox("Počet vlastností pre vysvetlenie", [3, 5, 6, 8, 10], index=2)

    if len(X_test_df) > 0:
        instance = X_test_df.iloc[selected_idx:selected_idx + 1]  # zachováme DataFrame s názvami stĺpcov
        exp = LimeTabularExplainer(
            training_data=X_train.values,
            training_labels=y_train,
            feature_names=X_train.columns.tolist(),
            discretize_continuous=True
        ).explain_instance(
            data_row=instance.iloc[0].values,
            predict_fn=lambda x: pipe.predict_proba(pd.DataFrame(x, columns=X_train.columns)),
            num_features=num_features
        )
        pred_class = pipe.predict(instance)[0]
        pred_label = le.inverse_transform([pred_class])[0]
        true_label = le.inverse_transform([y_test[selected_idx]])[0]

        st.subheader(f"Vysvetlenie pre vzorku s indexom {selected_idx}")
        st.markdown(f"**Skutočná trieda:** `{true_label}`  \n**Predikovaná trieda:** `{pred_label}`")

        with st.expander("Zobraziť textové vysvetlenie LIME"):
            explanation_list = exp.as_list()
            explanation_md = "\n".join([f"- **{item[0]}**: {item[1]:.4f}" for item in explanation_list])
            st.markdown(explanation_md)

        st.markdown("**Interaktívny graf LIME:**")
        fig_lime = exp.as_pyplot_figure()
        try:
            plotly_fig = tls.mpl_to_plotly(fig_lime)
            plotly_fig.update_layout(
                title="LIME Vizualizácia",
                title_font=dict(size=20, color="#0078AE"),
                xaxis_title="Vlastnosti",
                yaxis_title="Dopad",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(255,255,255,1)",
                transition_duration=500
            )
            st.plotly_chart(plotly_fig, use_container_width=True)
        except Exception as e:
            st.warning("Konverzia LIME grafu do Plotly sa nepodarila, zobrazujem matplotlib graf:")
            st.pyplot(fig_lime)
    else:
        st.warning("Testovacia množina je prázdna.")


@st.cache_resource(show_spinner=True)
def train_flaml_automl(X, y, time_budget=60):
    """
    Trénuje FLAML AutoML model so zadaným time_budgetom.
    Výsledný model je kešovaný pre zrýchlenie opakovaných výpočtov.
    """
    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": "flaml_oss_devs.log",
        "seed": 42
    }
    automl.fit(X, y, **settings)
    return automl


def page_automl():
    """
    Stránka pre FLAML AutoML s kešovaním výsledkov.
    """
    st.header("AutoML (FLAML) s kešovaním")
    df = load_data(DATA_FILE)
    df_features, y_encoded, le = prepare_data(df)
    if df_features is None:
        return

    if not FLAML_AVAILABLE or AutoML is None:
        st.info("FLAML nie je dostupné. Skontrolujte inštaláciu balíka flaml[automl].")
        return

    st.write("Nastavte time_budget (sekundy) pre FLAML:")
    tb = st.slider("time_budget", 10, 300, 60, 10)

    st.write("Rozdeľujeme dáta 80% / 20%...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    st.write("Spúšťame FLAML AutoML s kešovaním...")
    automl_model = train_flaml_automl(X_train, y_train, tb)

    y_pred = automl_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"[FLAML] Presnosť = {acc:.3f}")

    f1_ = f1_score(y_test, y_pred, average="macro", zero_division=0)
    st.metric("F1 (macro)", f"{f1_:.3f}")

    st.markdown("**AutoML výsledky sú cachované, aby sa znížila doba trénovania.**")


#############################################
# HLAVNÁ FUNKCIA A KONFIGURÁCIA
#############################################
def main():
    st.set_page_config(
        page_title="Open-Source Devs: Klasifikácia & Klastrovanie",
        layout="wide"
    )

    st.markdown("""
    <style>
    body {
        font-family: "Segoe UI", Tahoma, sans-serif;
        background: #f4f7fb;
        color: #2e2e2e;
        transition: background 0.5s ease;
    }
    .sidebar-content, .stDataFrame, .stTable {
        transition: all 0.5s ease;
    }
    .stButton>button:hover {
        background-color: #00A1C9 !important;
        transition: 0.3s;
    }
    h1, h2, h3 {
        color: #0078AE;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("Navigácia")
        st.markdown("""
        **O aplikácii:**
        - Načítanie a kešovanie datasetu
        - Detailný prehľad dát (vrátane rozdelenia rolí a ratingu)
        - Rôzne algoritmy klastrovania s interaktívnymi grafmi
        - RandomForest s hyperparametrickým ladením a 5-Fold CrossVal
        - Interpretácia pomocou LIME s interaktívnym grafom
        - AutoML s FLAML (kešovanie výsledkov)
        ---
        **Inštrukcie:**
        Vyberte si stránku pomocou rádia.
        """)
        page = st.radio(
            "Vyberte stránku:",
            ("Celkový prehľad datasetu", "Klastrovanie", "Klasifikácia", "Interpretácia (LIME)", "AutoML (FLAML)")
        )

    if page == "Celkový prehľad datasetu":
        page_dataset_overview()
    elif page == "Klastrovanie":
        page_data_and_clustering()
    elif page == "Klasifikácia":
        page_classification()
    elif page == "Interpretácia (LIME)":
        page_lime_interpretation()
    else:
        page_automl()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Aplikácia bola vytvorená pre demonštráciu pokročilých metód analýzy datasetov o open-source vývojároch.")
    st.markdown("<hr style='border:2px solid #0078AE'>", unsafe_allow_html=True)
    st.info("Hotovo. Ďakujeme za pozornosť!")


if __name__ == "__main__":
    main()
