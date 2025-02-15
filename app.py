# -*- coding: utf-8 -*-
"""
Autor: Mykyta Olym
Dátum: 2025
Popis:
Tento skript predstavuje rozšírenú demonštračnú aplikáciu v prostredí Streamlit.
Obsahuje nasledujúce funkcie:
 - Načítanie a predspracovanie datasetu (vrátane imputácie a enkódovania)
 - Dimenzionálnu redukciu s UMAP na vizualizáciu dát v 2D priestore
 - Klasifikáciu pomocou RandomForest s hyperparametrickým ladením cez RandomizedSearchCV a 5-Fold cross-validáciou
 - Výpočet rozšírených metrík (accuracy, precision, recall, F1, Balanced Accuracy, Cohen's Kappa, Matthews Corrcoef, ROC AUC, Log Loss, F2 score, Brier Score, Average Precision Score)
 - Interpretáciu modelu pomocou LIME s textovým a interaktívnym grafickým výstupom
 - Interaktívne vizualizácie pomocou Plotly (PCA scatter plot, Confusion Matrix, barplot rozdelenia rolí a rating)
 - Automatické ladenie modelu pomocou FLAML s kešovaním výsledkov pre zrýchlenie výpočtov

Poznámka:
Pre beh tejto aplikácie je potrebné nainštalovať knižnice:
 - streamlit
 - numpy, pandas
 - scikit-learn
 - plotly
 - lime
 - flaml[automl]
 - umap-learn
 - kaleido  # pre export grafov vo vektorovom formáte
Spustenie aplikácie: streamlit run app.py
"""

import time  # Modul pre meranie času
import numpy as np  # Knižnica pre prácu s číselnými dátami
import pandas as pd  # Knižnica pre prácu s dátovými rámcami (DataFrame)
import streamlit as st  # Framework pre rýchle vytváranie webových aplikácií

# Import pre interaktívne grafy pomocou Plotly
import plotly.express as px
import plotly.io as pio
import io  # Modul pre prácu s binárnymi dátami

# Nastavenie predvoleného šablónu pre grafy v Plotly
pio.templates.default = "plotly_white"

# Import LIME pre interpretáciu modelov
from lime.lime_tabular import LimeTabularExplainer

# Import základných tried a metód zo scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,  # Presnosť klasifikácie
    precision_score,  # Precision (presnosť) klasifikácie
    recall_score,  # Recall (citlivosť) klasifikácie
    f1_score,  # F1 skóre
    fbeta_score,  # F-beta skóre (tu najmä F2)
    brier_score_loss,  # Brier skóre pre odhad pravdepodobností
    average_precision_score,  # Priemerná precision pre viactriednu klasifikáciu
    confusion_matrix,  # Matica zámien (confusion matrix)
    classification_report,  # Textové zhrnutie metrík klasifikácie
    cohen_kappa_score,  # Cohen's Kappa pre hodnotenie zhody
    matthews_corrcoef,  # Matthews correlation coefficient
    balanced_accuracy_score,  # Vyvážená presnosť
    roc_auc_score,  # ROC AUC pre viactriedne modely
    log_loss  # Logaritmická strata
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Import FLAML (AutoML) – umožňuje automatické ladenie modelov
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    AutoML = None
    FLAML_AVAILABLE = False


#############################################
# POMOCNÁ FUNKCIA PRE EXPORT GRAFIKY VO VEKTOROVOM FORMÁTE
#############################################
def download_vector_figure(fig, filename="figure.svg"):
    """
    Funkcia pre prevod Plotly grafu na SVG a jeho následné stiahnutie.

    Vstup:
      - fig: Plotly graf (figure)
      - filename: názov súboru pre stiahnutie (predvolene "figure.svg")

    Výstup:
      - Tlačidlo pre stiahnutie vektorového súboru vo formáte SVG.

    Poznámka:
      Pre export do SVG je potrebné, aby bol nainštalovaný balík 'kaleido'.
      Táto verzia zabezpečuje, že ak sú rozmery grafu explicitne definované v layout-e (width, height),
      budú použité pri exporte, čím zostanú "originálne" rozmery grafu.
    """
    try:
        width = fig.layout.width if hasattr(fig.layout, "width") else None
        height = fig.layout.height if hasattr(fig.layout, "height") else None
        svg_bytes = fig.to_image(format="svg", width=width, height=height)
        st.download_button(
            label="Stiahnuť vektorovú grafiku (SVG)",
            data=svg_bytes,
            file_name=filename,
            mime="image/svg+xml"
        )
    except Exception as e:
        st.error("Nepodarilo sa exportovať graf: " + str(e))


#############################################
# DEFINÍCIE CESTY K DÁTAM
#############################################
# Upravte cestu k súboru s datasetom podľa potreby
DATA_FILE = "./Data/dataset_developers_metrics.csv"

#############################################
# PARAMETRE PRE KLASIFIKÁCIU
#############################################
# Parametre pre RandomForest klasifikátor, vrátane rôznych hodnôt pre hyperparametre
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

# Mapovanie hodnôt v stĺpci 'job' na kategórie (napr. "prof" pre profesionála, "skuseny" pre skúseného, atď.)
job_mapping = {
    "SE": "prof",
    "SSE": "skuseny",
    "SA": "skuseny",
    # Tu môžete pridať ďalšie mapovania podľa potreby
}

# Definícia váh pre jednotlivé numerické metriky – ovplyvňuje dôležitosť príznakov v modeli
feature_weights = {
    "followers": 1.0,
    "NoC": 1.0,
    "AB": 1.0,
    "NAB": 1.0,
    "CII": 1.5,
    "CNII": 1.5,
    "CE": 1.0,
    "NCE": 1.0,
    "INEI": 1.0,
    "IEI": 1.0,
    "AddLGM": 1.0,
    "DelLGM": 1.0,
    "ChurnLGM": 1.0,
    "NoMGM": 1.0,
    "AddLOC": 1.2,
    "DelLOC": 1.2,
    "churnLOC": 1.0,
    "AddF": 1.0,
    "DelF": 1.0,
    "AddSAM": 1.0,
    "DelSAM": 1.0,
    "ChurnSAM": 1.0,
    "DiP": 1.0,
    "ICT": 1.0
}


#############################################
# FUNKCIA PRE VÁHOVANIE PRÍZNAKOV (CUSTOM TRANSFORMER)
#############################################
def apply_feature_weights(X, weights):
    """
    Funkcia aplikuje váhovanie na jednotlivé stĺpce dát.
    Pre každý stĺpec, ktorý sa nachádza v zadanom slovníku 'weights',
    vynásobí hodnoty príslušnou váhou.

    Vstup:
      - X: pandas DataFrame s dátami
      - weights: slovník, kde kľúče sú názvy stĺpcov a hodnoty sú váhy

    Výstup:
      - X_weighted: DataFrame s aplikovanými váhami na príslušné stĺpce
    """
    X_weighted = X.copy()
    for col, weight in weights.items():
        if col in X_weighted.columns:
            X_weighted[col] = X_weighted[col].astype(float) * weight
    return X_weighted


# Vytvorenie transformeru pomocou FunctionTransformer, ktorý využije funkciu apply_feature_weights
feature_weighting_transformer = FunctionTransformer(lambda X: apply_feature_weights(X, feature_weights), validate=False)


#############################################
# FUNKCIE PRE NAČÍTANIE A PRÍPRAVU DÁT
#############################################
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    """
    Načíta dáta zo súboru CSV, zabezpečí konverziu textových stĺpcov na reťazce
    a vráti pandas DataFrame.

    Použitie kešovania (cache) zrýchľuje opakované načítanie dát.

    Vstup:
      - path: cesta k súboru CSV

    Výstup:
      - df: načítaný DataFrame s dátami
    """
    df = pd.read_csv(path)
    # Prevedenie všetkých stĺpcov typu 'object' na string pre kompatibilitu
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=True)
def prepare_data(df: pd.DataFrame):
    """
    Predspracuje dáta pre následnú analýzu.
    Kľúčové kroky:
      - Výber len potrebných numerických metrik (alebo všetkých dostupných)
      - Imputácia chýbajúcich hodnôt mediánom
      - Pre kategóriové stĺpce (napr. 'name' a 'project') sa vykoná Label Encoding
      - Pre stĺpec 'project' sa pripočíta frekvencia výskytu projektu
      - Dynamické mapovanie stĺpca 'job' na kategórie podľa job_mapping
      - Zakódovanie cieľovej premennej pomocou LabelEncoder

    Vstup:
      - df: pôvodný DataFrame s dátami

    Výstup:
      - df_features: DataFrame obsahujúci vybrané a predspracované príznaky
      - y_encoded: Zakódovaný vektor cieľových premenných
      - le: Inštancia LabelEncoder, ktorá sa použije na dekódovanie tried
    """
    # Zoznam metrik, ktoré chceme využiť (ak sú dostupné v dátach)
    metrics_list = ["followers", "NoC", "AB", "NAB", "CII", "CNII", "CE", "NCE",
                    "INEI", "IEI", "AddLGM", "DelLGM", "ChurnLGM", "NoMGM",
                    "AddLOC", "DelLOC", "churnLOC", "AddF", "DelF",
                    "AddSAM", "DelSAM", "ChurnSAM", "DiP", "ICT"]
    available_metrics = [col for col in metrics_list if col in df.columns]
    if available_metrics:
        df_metrics = df[available_metrics].copy()
    else:
        # Ak špecifikované metriky nie sú dostupné, použijeme všetky numerické stĺpce
        df_metrics = df.select_dtypes(include=[np.number]).copy()
    # Imputácia chýbajúcich hodnôt mediánom
    df_metrics = df_metrics.fillna(df_metrics.median())

    # Spracovanie kategóriových premenných – kódovanie mien a projektov
    df_cat = pd.DataFrame(index=df.index)
    if "name" in df.columns:
        le_name = LabelEncoder()
        df_cat["name_encoded"] = le_name.fit_transform(df["name"].astype(str))
    if "project" in df.columns:
        le_project = LabelEncoder()
        df_cat["project_encoded"] = le_project.fit_transform(df["project"].astype(str))
        # Výpočet frekvencie výskytu jednotlivých projektov
        project_freq = df["project"].value_counts()
        df_cat["project_freq"] = df["project"].map(project_freq)

    # Spojenie numerických a kategóriových dát do jedného DataFrame
    df_features = pd.concat([df_metrics, df_cat], axis=1)

    # Mapovanie cieľovej premennej 'job' na kategórie podľa preddefinovaného slovníka
    if "job" in df.columns:
        def dynamic_job_mapping(x):
            return job_mapping.get(str(x).upper(), "neznamy")
        df["job_class"] = df["job"].apply(dynamic_job_mapping)
    else:
        st.error("Chýba stĺpec 'job'.")
        return None, None, None

    # Zakódovanie cieľovej premennej do čísel pomocou LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(df["job_class"])
    return df_features, y_encoded, le


#############################################
# STRÁNKY A FUNKCIE PRE ANALÝZU A VIZUALIZÁCIU
#############################################
def page_dataset_overview():
    """
    Funkcia pre zobrazenie prehľadu datasetu:
      - Ukážka prvých niekoľkých riadkov dát
      - Rozšírené štatistiky (popis, medián, počet chýbajúcich hodnôt)
      - Zobrazenie korelačnej matice pre numerické stĺpce
      - Informácie o jednotlivých stĺpcoch (typ, počet unikátnych hodnôt, chýbajúce hodnoty)
      - Vizualizácia rozdelenia rolí vo forme bar grafu
    """
    st.header("Celkový prehľad datasetu")
    df = load_data(DATA_FILE)

    # Použitie tabov na oddelenie rôznych sekcií
    tab1, tab2, tab3 = st.tabs(["Ukážka dát", "Rozšírené štatistiky", "Informácie o stĺpcoch"])
    with tab1:
        st.subheader("Ukážka dát")
        num_rows = st.number_input("Počet riadkov na zobrazenie", min_value=5, max_value=50, value=10)
        st.dataframe(df.head(num_rows))
    with tab2:
        st.subheader("Rozšírené štatistiky pre číselné stĺpce")
        numeric_df = df.select_dtypes(include=[np.number])
        desc = numeric_df.describe().T
        # Pridanie mediánu a informácií o chýbajúcich hodnotách
        desc["median"] = numeric_df.median()
        desc["missing_count"] = df[numeric_df.columns].isna().sum()
        desc["missing_percent"] = (desc["missing_count"] / len(df)) * 100
        st.dataframe(desc)
        # Možnosť zobraziť korelačnú maticu
        if st.checkbox("Zobraziť korelačnú maticu"):
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Korelačná matica")
            st.plotly_chart(fig_corr, use_container_width=True)
            download_vector_figure(fig_corr, filename="korelacna_matica.svg")
    with tab3:
        st.subheader("Informácie o stĺpcoch")
        col_info = pd.DataFrame({
            "Typ": df.dtypes,
            "Chýbajúce hodnoty": df.isna().sum(),
            "Unikátne hodnoty": df.nunique()
        })
        st.table(col_info.astype(str))

    # Vizualizácia rozdelenia rolí (ak je dostupný stĺpec 'job_class')
    if "job_class" in df.columns:
        st.subheader("Rozdelenie rolí")
        job_counts = df["job_class"].value_counts().reset_index()
        job_counts.columns = ["job_class", "count"]
        fig_bar = px.bar(job_counts, x="job_class", y="count",
                         title="Rozdelenie vývojárov podľa úrovne",
                         color="job_class", text="count")
        fig_bar.update_layout(transition_duration=500)
        st.plotly_chart(fig_bar, use_container_width=True)
        download_vector_figure(fig_bar, filename="rozdelenie_rolí.svg")


def page_umap_visualization():
    """
    Funkcia demonštruje využitie UMAP na redukciu dimenzionality dát na 2D priestor.
    Používateľ si môže nastaviť parametre UMAP (počet susedov a minimálnu vzdialenosť)
    a výsledná vizualizácia zobrazuje body v 2D priestore, zafarbené podľa kategórie 'job_class'.
    """
    st.header("Vylepšená UMAP 2D Vizualizácia")
    df = load_data(DATA_FILE)
    df_features, y_encoded, le = prepare_data(df)
    if df_features is None:
        return

    st.write("Táto sekcia demonštruje využitie UMAP na redukciu dimenzionality dát na 2D priestor.")

    # Nastavenie parametrov UMAP
    n_neighbors = st.slider("Počet susedov (n_neighbors)", min_value=2, max_value=50, value=15, step=1)
    min_dist = st.slider("Minimálna vzdialenosť (min_dist)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    # Import knižnice UMAP
    try:
        import umap
    except ImportError:
        st.error("Knižnica umap-learn nie je nainštalovaná. Nainštalujte ju príkazom 'pip install umap-learn'.")
        return

    # Aplikácia UMAP na zníženie dimenzionality
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(df_features)

    # Príprava dát pre vizualizáciu – použitie zakódovanej cieľovej premennej pre farebné odlíšenie
    job_classes = le.inverse_transform(y_encoded)
    df_umap = pd.DataFrame({
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "JobClass": job_classes
    })

    fig = px.scatter(df_umap, x="UMAP1", y="UMAP2", color="JobClass",
                     title="UMAP 2D Vizualizácia", hover_data=["UMAP1", "UMAP2"])
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)
    download_vector_figure(fig, filename="umap_scatter.svg")


def page_classification():
    """
    Stránka pre klasifikáciu dát pomocou RandomForest.
    Obsahuje:
      - Vytvorenie pipeline so zahrnutým váhovaním príznakov, štandardizáciou a RandomForest klasifikátorom
      - Hyperparametrické ladenie pomocou RandomizedSearchCV (s 3-fold CV)
      - Vyhodnotenie modelu pomocou 5-Fold cross-validácie
      - Výpočet základných aj rozšírených metrík (Accuracy, Precision, Recall, F1, Balanced Accuracy, Cohen's Kappa, Matthews Corrcoef, F2 Score, Brier Score, Avg Precision, ROC AUC, Log Loss)
      - Vizualizácia ratingu vývojárov a matice zámien (confusion matrix)
    """
    st.header("Klasifikácia + Hyperparametrické ladenie + CrossVal")
    df = load_data(DATA_FILE)
    df_features, y_encoded, le = prepare_data(df)
    if df_features is None:
        return

    # Definícia pipeline, ktorá obsahuje:
    # 1. Váhovanie príznakov (custom transformer)
    # 2. Štandardizáciu dát (StandardScaler)
    # 3. RandomForest klasifikátor s pevne nastaveným random_state pre reprodukovateľnosť
    pipeline_steps = [
        ("feature_weighting", feature_weighting_transformer),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ]
    pipe = Pipeline(steps=pipeline_steps)

    # Definícia rozsahu hyperparametrov pre RandomForest pre použitie v RandomizedSearchCV
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
    # RandomizedSearchCV: testujeme 5 rôznych kombinácií hyperparametrov s 3-fold cross-validáciou
    search = RandomizedSearchCV(pipe, param_dist, n_iter=5, cv=3, scoring="f1_macro", random_state=42)
    search.fit(df_features, y_encoded)
    training_time = time.time() - start_time
    st.write("Čas ladenia: {:.2f} sekúnd".format(training_time))
    st.write("Najlepšie parametre:", search.best_params_)

    # Zobrazenie podrobných výsledkov testovaných modelov
    cv_results_df = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    st.subheader("Výsledky testovaných modelov")
    st.dataframe(cv_results_df)

    best_model = search.best_estimator_
    # Použitie 5-Fold cross-validácie pre vyhodnotenie modelu pomocou F1 (macro) skóre
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, df_features, y_encoded, cv=kf, scoring="f1_macro")
    st.write("5-Fold CrossVal F1 (macro):", cv_scores)
    st.write("Priemer F1:", np.mean(cv_scores), "±", np.std(cv_scores))

    # Predikcia na celom datasete (pre demonštračné účely)
    y_pred = best_model.predict(df_features)
    acc = accuracy_score(y_encoded, y_pred)
    prec = precision_score(y_encoded, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_encoded, y_pred, average="macro", zero_division=0)
    f1_ = f1_score(y_encoded, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_encoded, y_pred)
    kappa = cohen_kappa_score(y_encoded, y_pred)
    mcc = matthews_corrcoef(y_encoded, y_pred)

    # Výpočet dodatočných metrík
    f2 = fbeta_score(y_encoded, y_pred, beta=2, average="macro", zero_division=0)
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(df_features)
        # Výpočet Brier skóre pre viactriednu klasifikáciu:
        if len(np.unique(y_encoded)) == 2:
            brier = brier_score_loss(y_encoded, y_proba.max(axis=1))
        else:
            classes = np.arange(len(le.classes_))
            y_true_bin = label_binarize(y_encoded, classes=classes)
            brier = np.mean(np.sum((y_true_bin - y_proba) ** 2, axis=1))
        avg_precision = average_precision_score(y_encoded, y_proba, average="macro")
    else:
        brier = None
        avg_precision = None

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

    # Zobrazenie základných metrík
    st.markdown("**Základné metriky**:")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision (macro)", f"{prec:.3f}")
    col3.metric("Recall (macro)", f"{rec:.3f}")
    col4.metric("F1 (macro)", f"{f1_:.3f}")

    # Zobrazenie rozšírených metrík
    st.markdown("**Rozšírené metriky**:")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Balanced Accuracy", f"{bal_acc:.3f}")
    col6.metric("Cohen's Kappa", f"{kappa:.3f}")
    col7.metric("Matthews Corrcoef", f"{mcc:.3f}")
    col8.metric("F2 Score", f"{f2:.3f}")
    if brier is not None:
        st.metric("Brier Score", f"{brier:.3f}")
    if avg_precision is not None:
        st.metric("Avg Precision Score", f"{avg_precision:.3f}")
    if additional_metrics:
        for metric_name, value in additional_metrics.items():
            st.metric(metric_name, f"{value:.3f}")

    # Výpočet ratingu – využíva pravdepodobnosť pre kategóriu "prof"
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(df_features)
        prof_index = np.where(le.classes_ == "prof")[0]
        if len(prof_index) > 0:
            prof_index = prof_index[0]
            ratings = y_proba[:, prof_index] * 100  # Konverzia pravdepodobnosti na skóre 0-100
            rating_df = pd.DataFrame({"Developer": np.arange(len(ratings)), "Rating": ratings})
            rating_df = rating_df.sort_values("Rating", ascending=False)
            st.subheader("Rating vývojárov (0-100)")
            fig_rating = px.bar(rating_df, x="Developer", y="Rating",
                                title="Rating vývojárov podľa pravdepodobnosti pre 'prof'",
                                text="Rating")
            fig_rating.update_layout(transition_duration=500)
            st.plotly_chart(fig_rating, use_container_width=True)
            download_vector_figure(fig_rating, filename="rating_vyvojarov.svg")

    # Zobrazenie matice zámien (confusion matrix)
    cm = confusion_matrix(y_encoded, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                       labels={"x": "Predikovaná trieda", "y": "Skutočná trieda"},
                       x=le.classes_, y=le.classes_,
                       title="Confusion Matrix")
    fig_cm.update_layout(transition_duration=500)
    st.plotly_chart(fig_cm, use_container_width=True)
    download_vector_figure(fig_cm, filename="confusion_matrix.svg")
    st.markdown("**Classification Report:**")
    cr = classification_report(y_encoded, y_pred, target_names=le.classes_, zero_division=0)
    st.text(cr)


def page_lime_interpretation():
    """
    Stránka pre interpretáciu predikcií pomocou LIME.
    Používateľ si môže zvoliť konkrétny index vzorky zo testovacej množiny,
    pre ktorú sa vygeneruje vysvetlenie modelovej predikcie.

    Kroky:
      - Rozdelenie dát na trénovaciu a testovaciu množinu
      - Natrénovanie modelu (RandomForest) na trénovacích dátach
      - Použitie LIME na vysvetlenie predikcie pre vybranú vzorku
      - Zobrazenie textového vysvetlenia aj interaktívneho grafu s prehľadným usporiadaním
    """
    st.header("Interpretácia predikcií s LIME")
    df = load_data(DATA_FILE)
    df_features, y_encoded, le = prepare_data(df)
    if df_features is None:
        return
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    X_test_df = X_test.copy()
    st.write("Vyberte index vzorky pre vysvetlenie (0 - {}):".format(len(X_test_df) - 1))
    selected_idx = st.slider("Index vzorky", 0, len(X_test_df) - 1, 0)
    num_features = st.selectbox("Počet vlastností pre vysvetlenie", [3, 5, 6, 8, 10], index=2)
    if len(X_test_df) > 0:
        instance = X_test_df.iloc[selected_idx:selected_idx + 1]
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            training_labels=y_train,
            feature_names=X_train.columns.tolist(),
            discretize_continuous=True
        )
        exp = explainer.explain_instance(
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
        # Vytvorenie prehľadného interaktívneho grafu s využitím Plotly
        explanation_list = exp.as_list()
        explanation_df = pd.DataFrame(explanation_list, columns=["Feature", "Contribution"])
        # Zoradenie podľa príspevku pre lepšiu prehľadnosť (od najnižšieho po najvyšší)
        explanation_df = explanation_df.sort_values(by="Contribution", ascending=True)
        fig_lime = px.bar(
            explanation_df,
            x="Contribution",
            y="Feature",
            orientation='h',
            title="LIME Explanation: Príspevky príznakov",
            labels={"Contribution": "Príspevok k predikcii", "Feature": "Príznak"},
            color="Contribution",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_lime, use_container_width=True)
        download_vector_figure(fig_lime, filename="lime_explanation.svg")
    else:
        st.warning("Testovacia množina je prázdna.")


@st.cache_resource(show_spinner=True)
def train_flaml_automl(X, y, time_budget=60):
    """
    Funkcia na natrénovanie modelu pomocou FLAML AutoML.
    Využíva cache, takže pri opakovanom spustení s rovnakými dátami sa model natrénuje len raz.

    Vstup:
      - X: Vstupné príznaky
      - y: Cieľová premenná
      - time_budget: Časový limit v sekundách pre ladenie modelu

    Výstup:
      - automl: Natrénovaný FLAML AutoML model
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
    Stránka demonštrujúca použitie FLAML AutoML na ladenie modelu.
    Dáta sa rozdelia na trénovaciu (80%) a testovaciu (20%) množinu.
    Výsledný model sa otestuje na testovacej množine a zobrazí sa presnosť a F1 skóre.
    Kešovanie znižuje dobu opakovaného trénovania.
    Navyše, sú zobrazené detaily o vybranej modeli.
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

    # Zobrazenie detailov o vybranej modeli FLAML
    st.subheader("Vybraná modelová konfigurácia FLAML")
    st.write("Best Estimator:", automl_model.best_estimator)
    st.write("Best Config:", automl_model.best_config)
    st.markdown("**AutoML výsledky sú cachované, aby sa znížila doba trénovania.**")


#############################################
# HLAVNÁ FUNKCIA A KONFIGURÁCIA APLIKÁCIE
#############################################
def main():
    """
    Hlavná funkcia aplikácie.
    Nastavuje základné vlastnosti stránky, navigačný sidebar a spúšťa vybranú sekciu (stránku).

    Sekcie obsahujú:
      - Celkový prehľad datasetu
      - Dimenzionálnu redukciu dát pomocou UMAP
      - Klasifikáciu s hyperparametrickým ladením a 5-Fold CrossVal
      - Interpretáciu pomocou LIME
      - Automatické ladenie pomocou FLAML
    """
    st.set_page_config(
        page_title="Open-Source Devs: Klasifikácia & Dimenzionálna redukcia",
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
        - Dimenzionálna redukcia dát pomocou UMAP (2D vizualizácia)
        - RandomForest s hyperparametrickým ladením a 5-Fold CrossVal
        - Interpretácia pomocou LIME s interaktívnym grafom
        - AutoML s FLAML (kešovanie výsledkov)
        ---
        **Inštrukcie:**
        Vyberte si stránku pomocou rádia.
        """)
        page = st.radio(
            "Vyberte stránku:",
            ("Celkový prehľad datasetu", "Dimenzionálna redukcia (UMAP)", "Klasifikácia", "Interpretácia (LIME)",
             "AutoML (FLAML)")
        )
    if page == "Celkový prehľad datasetu":
        page_dataset_overview()
    elif page == "Dimenzionálna redukcia (UMAP)":
        page_umap_visualization()
    elif page == "Klasifikácia":
        page_classification()
    elif page == "Interpretácia (LIME)":
        page_lime_interpretation()
    else:
        page_automl()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Aplikácia bola vytvorená pre demonštráciu pokročilých metód analýzy datasetov o open-source vývojároch."
    )
    st.markdown("<hr style='border:2px solid #0078AE'>", unsafe_allow_html=True)
    st.info("Hotovo. Ďakujeme za pozornosť!")


# Spustenie aplikácie
if __name__ == "__main__":
    main()
