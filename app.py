import html
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "notebooks" / "outputs"
LABELS = {0: "Negative", 1: "Positive"}
FEATURES = {"TF-IDF": "tfidf", "Bag of words": "bow"}
MODELS = {
    "Naive Bayes": {
        "bow": "nb_bow_model.joblib",
        "tfidf": "nb_tfidf_model.joblib",
    },
    "Logistic regression": {
        "bow": "lr_bow_model.joblib",
        "tfidf": "lr_tfidf_model.joblib",
    },
}
EXAMPLES = {
    "Write your own": "",
    "Praise": (
        "The coffee tastes fresh and rich. Great flavor and excellent value."
    ),
    "Complaint": (
        "The crackers arrived stale and broken. The flavor was awful."
    ),
    "Mixed opinion": (
        "The tea smells wonderful, but the taste is weak and the price is too high."
    ),
}


def clean_text(text):
    # Match feature_engineering.py for new reviews.
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip()


def file_versions(paths):
    # Refresh the cache when a model or data file changes.
    versions = []
    for path in paths:
        info = path.stat()
        versions.append((str(path), info.st_mtime_ns, info.st_size))
    return tuple(versions)


@st.cache_resource(show_spinner=False)
def load_assets(versions):
    vectorizers = {
        feature: joblib.load(OUTPUTS / f"{feature}_vectorizer.joblib")
        for feature in FEATURES.values()
    }

    models = {}
    for name, filenames in MODELS.items():
        for feature, filename in filenames.items():
            model = joblib.load(ROOT / "models" / filename)

            if model.n_features_in_ != len(vectorizers[feature].vocabulary_):
                raise ValueError(
                    f"{filename} does not match its vectorizer."
                )

            if list(model.classes_) != [0, 1]:
                raise ValueError(
                    f"{filename} has an unexpected label mapping."
                )

            models[name, feature] = model

    return vectorizers, models


@st.cache_data(show_spinner=False)
def load_labels(versions):
    labels = {}

    for split in ("train", "test"):
        values = pd.read_csv(
            OUTPUTS / f"{split}.csv",
            usecols=["Sentiment"],
        )["Sentiment"]

        if values.empty or not values.isin([0, 1]).all():
            raise ValueError(
                f"{split}.csv must contain binary sentiment labels."
            )

        labels[split] = values.astype(int)

    return labels


@st.cache_data(show_spinner=False)
def evaluate_models(asset_versions, data_versions):
    _, models = load_assets(asset_versions)
    labels = load_labels(data_versions)
    actual = labels["test"]
    rows, details = [], {}

    for representation, feature in FEATURES.items():
        matrix = load_npz(OUTPUTS / f"X_test_{feature}.npz")

        if matrix.shape[0] != len(actual):
            raise ValueError(
                f"X_test_{feature}.npz does not match test.csv."
            )

        for name in MODELS:
            model = models[name, feature]

            if matrix.shape[1] != model.n_features_in_:
                raise ValueError(
                    f"Test features do not match {name} ({representation})."
                )

            predicted = model.predict(matrix)
            precision, recall, f1, support = precision_recall_fscore_support(
                actual,
                predicted,
                labels=[0, 1],
                zero_division=0,
            )

            key = f"{name} / {representation}"

            rows.append({
                "Model": name,
                "Features": representation,
                "Accuracy": accuracy_score(actual, predicted),
                "Macro F1": f1_score(
                    actual,
                    predicted,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                ),
                "Negative recall": recall[0],
                "Negative F1": f1[0],
            })

            details[key] = {
                "matrix": pd.DataFrame(
                    confusion_matrix(actual, predicted, labels=[0, 1]),
                    index=["Actual negative", "Actual positive"],
                    columns=["Predicted negative", "Predicted positive"],
                ),
                "classes": pd.DataFrame({
                    "Class": ["Negative", "Positive"],
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Reviews": support,
                }),
            }

    return pd.DataFrame(rows), details


def use_example():
    st.session_state["review_text"] = EXAMPLES[
        st.session_state["example"]
    ]
    st.session_state.pop("analysis", None)


def show_error(message, error):
    st.error(message)
    with st.expander("Details"):
        st.code(str(error), language="text")


st.set_page_config(
    page_title="Food Review Sentiment | Group Alpha",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1160px;
        padding-top: 2.5rem;
    }
    h1 {
        letter-spacing: -0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("THE OHIO STATE UNIVERSITY  /  CSE 3521  /  GROUP ALPHA")
st.title("Food Review Sentiment")
st.write(
    "Compare how Naive Bayes and logistic regression interpret "
    "the same food review."
)

asset_paths = [
    OUTPUTS / f"{feature}_vectorizer.joblib"
    for feature in FEATURES.values()
]
asset_paths += [
    ROOT / "models" / filename
    for files in MODELS.values()
    for filename in files.values()
]

missing = [
    str(path.relative_to(ROOT))
    for path in asset_paths
    if not path.is_file()
]

if missing:
    st.error("The following model or vectorizer files are missing:")
    st.code("\n".join(missing), language="text")
    st.write(
        "If the missing files are the logistic-regression models, "
        "run this from the project folder:"
    )
    st.code(
        "cd notebooks\npython train_logistic_regression.py\ncd ..",
        language="bash",
    )
    st.stop()

try:
    asset_stamp = file_versions(asset_paths)
    vectorizers, models = load_assets(asset_stamp)
except Exception as error:
    show_error("The saved models could not be loaded.", error)
    st.stop()

review_tab, performance_tab, dataset_tab = st.tabs([
    "Review analysis",
    "Test performance",
    "Dataset",
])

with review_tab:
    st.subheader("Analyze a review")

    st.selectbox(
        "Example review",
        list(EXAMPLES),
        key="example",
        on_change=use_example,
    )
    st.caption(
        "Examples are written for this demo; they are not test-set reviews."
    )

    with st.form("review_form"):
        representation = st.radio(
            "Text representation",
            list(FEATURES),
            horizontal=True,
        )
        review = st.text_area(
            "Review text",
            key="review_text",
            height=145,
            max_chars=5000,
            placeholder=(
                "Describe the taste, quality, packaging, "
                "or value of a food product."
            ),
        )
        submitted = st.form_submit_button(
            "Analyze review",
            type="primary",
        )

    if submitted:
        st.session_state.pop("analysis", None)
        cleaned = clean_text(review)

        if not cleaned:
            st.warning("Enter a review containing text.")
        else:
            feature = FEATURES[representation]
            matrix = vectorizers[feature].transform([cleaned])

            if matrix.nnz == 0:
                st.warning(
                    "No words in this review match the model vocabulary. "
                    "Try a fuller food review."
                )
            else:
                results = []

                for name in MODELS:
                    model = models[name, feature]
                    positive_column = list(model.classes_).index(1)

                    results.append({
                        "name": name,
                        "label": LABELS[int(model.predict(matrix)[0])],
                        "positive_probability": float(
                            model.predict_proba(matrix)[0, positive_column]
                        ),
                    })

                st.session_state["analysis"] = {
                    "review": review,
                    "representation": representation,
                    "results": results,
                    "versions": asset_stamp,
                }

    analysis = st.session_state.get("analysis")

    if analysis and analysis["versions"] != asset_stamp:
        st.session_state.pop("analysis", None)
        analysis = None

    if analysis:
        st.subheader("Model comparison")
        st.caption(
            f"Last submitted review / {analysis['representation']}"
        )

        for column, result in zip(st.columns(2), analysis["results"]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{result['name']}**")
                    st.metric("Predicted sentiment", result["label"])

                    probability = result["positive_probability"]
                    st.write(
                        "Estimated probability of positive sentiment: "
                        f"**{probability:.1%}**"
                    )
                    st.progress(probability)

        st.caption(
            "Probabilities are uncalibrated model estimates, "
            "not test accuracy."
        )

        with st.expander("Submitted review"):
            st.write(analysis["review"])

with performance_tab:
    st.subheader("Performance on the saved test split")
    st.write(
        "All four configurations are evaluated on the same held-out reviews."
    )

    try:
        evaluation_paths = [
            OUTPUTS / name
            for name in (
                "train.csv",
                "test.csv",
                "X_test_bow.npz",
                "X_test_tfidf.npz",
            )
        ]
        data_stamp = file_versions(evaluation_paths)

        with st.spinner("Evaluating saved models..."):
            summary, details = evaluate_models(asset_stamp, data_stamp)
            labels = load_labels(data_stamp)

        majority = int(labels["train"].mode().iloc[0])
        baseline = float((labels["test"] == majority).mean())

        st.caption(
            f"{len(labels['test']):,} test reviews. "
            f"Always predicting {LABELS[majority].lower()} "
            f"scores {baseline:.1%} accuracy; this baseline uses "
            "the training set's majority label."
        )

        numeric_columns = [
            "Accuracy",
            "Macro F1",
            "Negative recall",
            "Negative F1",
        ]

        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in numeric_columns
            },
        )

        st.caption(
            "Scores use a 0-1 scale. Macro F1 weights both classes equally. "
            "Negative recall measures how many negative reviews "
            "the model identifies."
        )

        selected = st.selectbox(
            "Inspect a configuration",
            list(details),
        )

        st.markdown("#### Confusion matrix")
        st.table(details[selected]["matrix"])

        st.markdown("#### Results by class")
        st.dataframe(
            details[selected]["classes"],
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in ("Precision", "Recall", "F1")
            },
        )

    except Exception as error:
        show_error(
            "Test-set evaluation is unavailable. "
            "Review analysis can still be used.",
            error,
        )

with dataset_tab:
    st.subheader("Dataset and preprocessing")

    try:
        label_paths = [
            OUTPUTS / "train.csv",
            OUTPUTS / "test.csv",
        ]
        labels = load_labels(file_versions(label_paths))
        train_count = len(labels["train"])
        test_count = len(labels["test"])

        total_column, train_column, test_column = st.columns(3)
        total_column.metric(
            "Reviews in saved splits", f"{train_count + test_count:,}"
        )
        train_column.metric("Training reviews", f"{train_count:,}")
        test_column.metric("Test reviews", f"{test_count:,}")
        rows = []
        for split, values in labels.items():
            rows.append({
                "Split": split.title(),
                "Negative": int((values == 0).sum()),
                "Positive": int((values == 1).sum()),
                "Negative share": f"{(values == 0).mean():.1%}",
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

    except Exception as error:
        show_error(
            "Dataset counts could not be read from the saved splits.",
            error,
        )

    st.markdown("#### Label definition")
    st.write(
        "Ratings of 4 or 5 stars are positive; ratings of 1 or 2 stars "
        "are negative. Three-star reviews are excluded."
    )

    st.markdown("#### Preparation")
    st.write(
        "The pipeline removes missing text or ratings and duplicate "
        "user-product-text records, decodes HTML entities, removes "
        "HTML tags, and drops empty reviews. It creates a stratified "
        "80/20 split with random seed 42."
    )
    st.write(
        "Both vectorizers are fitted on training reviews only, remove "
        "English stop words, and retain up to 10,000 terms. Bag of words "
        "uses word counts; TF-IDF weights terms by their frequency "
        "within a review and across the training set."
    )

    st.caption(
        "Saved vocabulary sizes: "
        f"bag of words {len(vectorizers['bow'].vocabulary_):,}; "
        f"TF-IDF {len(vectorizers['tfidf'].vocabulary_):,}."
    )

    st.markdown("#### Scope")
    st.write(
        "The app predicts positive or negative sentiment. It has no "
        "neutral class. These models use individual word features, "
        "so mixed opinions, sarcasm, and negation can be difficult "
        "to interpret. English stop-word filtering also removes "
        "some negation terms."
    )