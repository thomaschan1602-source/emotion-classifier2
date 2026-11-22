"""CBS4702 Final Project - Interactive Emotion Classification Web App (Sec. 6)

Built with Streamlit and AI-assisted "vibe coding" style development.
Users can:
  - Enter their own English text and obtain an emotion prediction with confidence.
  - Select and compare any baseline vs improved model.
  - Explore dataset/model visualizations (emotion distribution, confusion matrices,
    per-class F1 comparison, model comparison).
  - Browse example predictions, including real error cases from the test set.

Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from scipy import sparse

# ----------------------------------------------------------------------
# Paths & constants
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
FIG_DIR = os.path.join(BASE, "figures")
RES_DIR = os.path.join(BASE, "results")

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Model registry: key -> (display name, type, pipeline type)
MODEL_REGISTRY = {
    "bow_lr": ("BoW + Logistic Regression (baseline)", "Baseline"),
    "tfidf_lr": ("TF-IDF + Logistic Regression (baseline)", "Baseline"),
    "tfidf_nb": ("TF-IDF + Multinomial NB (baseline)", "Baseline"),
    "bow_lr_balanced": ("BoW + LR, balanced weights (improved)", "Improved"),
    "tfidf_wordchar_lr": ("TF-IDF word+char n-gram + LR (improved)", "Improved"),
    "tfidf_lsvc": ("TF-IDF + Linear SVC (improved)", "Improved"),
}

st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="wide")


# ----------------------------------------------------------------------
# Model loading (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all six trained pipelines from disk."""
    models = {}
    models["bow_lr"] = joblib.load(os.path.join(MODEL_DIR, "bow_lr.joblib"))
    models["bow_lr_balanced"] = joblib.load(os.path.join(MODEL_DIR, "bow_lr_balanced.joblib"))
    models["tfidf_lr"] = joblib.load(os.path.join(MODEL_DIR, "tfidf_lr.joblib"))
    models["tfidf_nb"] = joblib.load(os.path.join(MODEL_DIR, "tfidf_nb.joblib"))
    models["tfidf_wordchar_lr"] = joblib.load(os.path.join(MODEL_DIR, "tfidf_wordchar_lr.joblib"))
    models["tfidf_lsvc"] = joblib.load(os.path.join(MODEL_DIR, "tfidf_lsvc.joblib"))
    return models


@st.cache_data
def load_error_cases():
    path = os.path.join(RES_DIR, "error_cases.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_summary():
    path = os.path.join(RES_DIR, "final_model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_text(model_key, text):
    """Return (predicted_label_index, confidence, proba_vector)."""
    models = load_models()
    pipe = models[model_key]
    if model_key == "tfidf_wordchar_lr":
        word_vec, char_vec, clf = pipe
        Xw = word_vec.transform([text])
        Xc = char_vec.transform([text])
        X = sparse.hstack([Xw, Xc]).tocsr()
    else:
        vec, clf = pipe
        X = vec.transform([text])

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
    else:
        # LinearSVC: approximate confidence via softmax of decision function
        scores = clf.decision_function(X)[0]
        proba = _softmax(scores)

    pred = int(np.argmax(proba))
    conf = float(proba[pred])
    return pred, conf, proba


def validate_text(text):
    """Return (ok, message)."""
    t = (text or "").strip()
    if not t:
        return False, "The input is empty. Please type an English sentence."
    if not any(c.isalpha() for c in t):
        return False, "The input does not contain any letters. Please enter an English sentence."
    return True, ""


# ----------------------------------------------------------------------
# Visualizations
# ----------------------------------------------------------------------
def plot_per_class_f1(model_key):
    """Draw per-class F1 bar chart from saved results."""
    prefix_map = {
        "bow_lr": "bow_lr",
        "bow_lr_balanced": "improved_bow_balanced",
        "tfidf_lr": "tfidf_lr",
        "tfidf_nb": "tfidf_nb",
        "tfidf_wordchar_lr": "improved_wordchar_lr",
        "tfidf_lsvc": "improved_lsvc",
    }
    path = os.path.join(RES_DIR, f"{prefix_map[model_key]}_per_class.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["emotion"].isin(EMOTIONS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    ax.bar(df["emotion"], df["f1"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(f"Per-class F1 - {MODEL_REGISTRY[model_key][0]}")
    for i, v in enumerate(df["f1"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    return fig


def plot_model_comparison():
    df = load_model_summary()
    if df.empty:
        return None
    df = df.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    ax.barh(y, df["macro_f1"], color="#55A868")
    ax.set_yticks(y, [m.replace(" + ", " + ") for m in df["model"]], fontsize=8)
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("Macro-F1 (test)")
    ax.set_title("All model comparison (macro-F1, test set)")
    for i, (a, f) in enumerate(zip(y, df["macro_f1"])):
        ax.text(f + 0.005, i, f"{f:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def main():
    st.title("🎭 Emotion Classification App")
    st.markdown(
        "CBS4702 Final Project — classify short English texts into six emotions: "
        "**sadness**, **joy**, **love**, **anger**, **fear**, **surprise**."
    )

    models = load_models()
    with st.sidebar:
        st.header("Model selection")
        model_key = st.selectbox(
            "Choose a model",
            list(MODEL_REGISTRY.keys()),
            format_func=lambda k: f"{MODEL_REGISTRY[k][1]} | {MODEL_REGISTRY[k][0]}",
            index=3,  # default: best model (bow_lr_balanced)
        )
        st.info(
            f"**Type:** {MODEL_REGISTRY[model_key][1]}\n\n"
            f"Select a baseline (BoW+LR / TF-IDF+LR / NB) and an improved "
            f"(balanced weights / word+char n-gram / SVC) model to compare them."
        )

    # ---------------- Prediction panel ----------------
    st.header("1. Predict an emotion")
    example_cols = st.columns(4)
    default_text = "i feel so sad and lonely today"
    example_pool = {
        "Happy": "i am so happy and excited about the party tonight",
        "Angry": "i am really angry because he lied to me again",
        "Scared": "i am scared of the dark and always check under my bed",
        "Surprised": "wow i cant believe you planned all of this for me",
    }
    if example_cols[0].button("😊 Happy"):
        default_text = example_pool["Happy"]
    if example_cols[1].button("😡 Angry"):
        default_text = example_pool["Angry"]
    if example_cols[2].button("😨 Scared"):
        default_text = example_pool["Scared"]
    if example_cols[3].button("😲 Surprised"):
        default_text = example_pool["Surprised"]

    user_text = st.text_area("Enter English text:", value=default_text, height=100,
                             placeholder="e.g. I feel so sad and lonely today")

    if st.button("Predict", type="primary"):
        ok, msg = validate_text(user_text)
        if not ok:
            st.warning(msg)
        else:
            pred, conf, proba = predict_text(model_key, user_text)
            st.markdown(f"### Predicted emotion: **{EMOTIONS[pred]}**")
            st.markdown(f"Confidence: **{conf:.2%}** (model: {MODEL_REGISTRY[model_key][0]})")

            # probability bar chart
            proba_df = pd.DataFrame({"emotion": EMOTIONS, "probability": proba})
            proba_df = proba_df.sort_values("probability", ascending=False)
            fig2, ax2 = plt.subplots(figsize=(7, 3.2))
            ax2.bar(proba_df["emotion"], proba_df["probability"], color="#4C72B0")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Probability")
            ax2.set_title("Class probabilities")
            for i, v in enumerate(proba_df["probability"]):
                ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

    st.markdown("---")
    # ---------------- Visualizations ----------------
    st.header("2. Dataset & model visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Emotion distribution (train set)")
        st.image(os.path.join(FIG_DIR, "01_emotion_distribution.png"))
    with col2:
        st.subheader("Best model confusion matrix (test set)")
        st.image(os.path.join(FIG_DIR, "04_best_model_confusion.png"))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Per-class F1 of selected model")
        f = plot_per_class_f1(model_key)
        if f is not None:
            st.pyplot(f)
        else:
            st.info("Per-class results not available for this model.")
    with col4:
        st.subheader("Model comparison (macro-F1)")
        f2 = plot_model_comparison()
        if f2 is not None:
            st.pyplot(f2)
        else:
            st.info("Comparison table not found.")

    st.markdown("---")
    # ---------------- Error examples ----------------
    st.header("3. Example predictions & error cases")
    err = load_error_cases()
    if not err.empty:
        st.markdown(
            "The table below shows **15 real mispredictions** of the best model "
            "(BoW + LR balanced) on the test set. Use these to understand model limitations."
        )
        st.dataframe(err[["text", "true_label", "pred_label", "explanation"]],
                     use_container_width=True, height=350)
    else:
        st.info("error_cases.csv not found; example error cases are unavailable.")

    # Quick live predictions on error examples
    st.markdown("**Try predicting an error example yourself:**")
    if not err.empty:
        sel = st.selectbox("Pick an error text", err["text"].tolist())
        ok, msg = validate_text(sel)
        if ok:
            pred, conf, proba = predict_text(model_key, sel)
            row = err[err["text"] == sel].iloc[0]
            st.write(
                f"True label: **{row['true_label']}**  |  "
                f"Model predicts: **{EMOTIONS[pred]}** ({conf:.1%})"
            )

    st.markdown("---")
    st.caption(
        "All models were trained on dair-ai/emotion (16,000 train / 2,000 validation / 2,000 test). "
        "Confidence for Linear SVC is a softmax approximation of the decision function."
    )


if __name__ == "__main__":
    main()
