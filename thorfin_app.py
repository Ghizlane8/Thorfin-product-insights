"""
Thorfin Product Insights – Application IA avec Streamlit

Objectifs :
- Identifier les meilleurs / pires produits
- Explorer les performances d’un produit donné (notes, prix, avis)
- Fournir une fonctionnalité IA : analyse de sentiment supervisée sur les avis clients
- Aider à la décision : recommandations produits & produits similaires
- Analyser les performances par pays / langue (inspiré de Thorfin Product Pulse)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity  # pour produits similaires


# ===================================================
# 1. CONFIGURATION STREAMLIT
# ===================================================

st.set_page_config(
    page_title="Thorfin Product Insights",
    page_icon="📈",
    layout="wide",
)

# 🎨 Thème clair personnalisé
st.markdown(
    """
    <style>
    /* Fond principal clair */
    .main {
        background-color: #f5f5f5;
    }

    /* Conteneur central légèrement ombré */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Cartes de métriques plus claires */
    .stMetric {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Sidebar claire */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }

    /* Titres légèrement recolorés */
    h1, h2, h3, h4 {
        color: #1f2933;
    }

    /* Texte général */
    .stMarkdown, .stText, p {
        color: #111827;
    }

    /* Cartes leaders & laggards */
    .leader-card {
        background-color: #ffffff;
        border-radius: 0.9rem;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.8rem;
    }

    .leader-card.bad {
        border-left-color: #ef4444;
    }

    .leader-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }

    .leader-product {
        font-weight: 600;
        font-size: 0.98rem;
        color: #111827;
        margin-bottom: 0.15rem;
    }

    .leader-sub {
        font-size: 0.85rem;
        color: #4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================================================
# 2. CONFIG GLOBALE & CHARGEMENT DES DONNÉES
# ===================================================

DATA_PATH = r"D:\AI INSTITUTE\ML youssef\Teste\thorfin_reviews_electronics_home_appliance.csv"

COL_PRODUCT = "product"
COL_DESCRIPTION = "product_description"
COL_PRICE = "price"
COL_RATING = "rating"
COL_REVIEW_TEXT = "review_text"
COL_LANGUAGE = "review_language"  # utilisé pour le mapping langue → pays

LANGUAGE_COUNTRY_MAP: Dict[str, str] = {
    "en": "United States",
    "fr": "France",
    "es": "Spain",
    "de": "Germany",
    "ar": "United Arab Emirates",
    "it": "Italy",
    "pt": "Portugal",
    "zh": "China",
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected_cols = [COL_PRODUCT, COL_DESCRIPTION, COL_PRICE, COL_RATING, COL_REVIEW_TEXT]
    for col in expected_cols:
        if col not in df.columns:
            st.warning(
                f"La colonne attendue '{col}' n'existe pas dans le CSV. "
                f"Adapte COL_... dans le code si nécessaire."
            )

    if COL_RATING in df.columns:
        df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")

    if COL_PRICE in df.columns:
        df[COL_PRICE] = pd.to_numeric(df[COL_PRICE], errors="coerce")

    for col in [COL_PRODUCT, COL_RATING, COL_REVIEW_TEXT]:
        if col in df.columns:
            df = df[df[col].notna()]

    df = df[(df[COL_RATING] >= 1) & (df[COL_RATING] <= 5)]

    df.reset_index(drop=True, inplace=True)
    return df


df: pd.DataFrame = load_data(DATA_PATH)


# ===================================================
# 3. ENRICHISSEMENT : SENTIMENT + PAYS / LANGUE
# ===================================================

def enrich_with_country_and_flags(df_source: pd.DataFrame) -> pd.DataFrame:
    enriched = df_source.copy()

    if COL_LANGUAGE in enriched.columns:
        enriched["country"] = (
            enriched[COL_LANGUAGE]
            .map(LANGUAGE_COUNTRY_MAP)
            .fillna("Unknown")
        )
    elif "country" in enriched.columns:
        enriched["country"] = enriched["country"].fillna("Unknown")
    else:
        enriched["country"] = "Unknown"

    if COL_RATING in enriched.columns:
        enriched["positive_review"] = enriched[COL_RATING].astype(float) >= 4
        enriched["negative_review"] = enriched[COL_RATING].astype(float) <= 2
    else:
        enriched["positive_review"] = False
        enriched["negative_review"] = False

    return enriched


@st.cache_data
def compute_product_stats(df_source: pd.DataFrame) -> pd.DataFrame:
    group = df_source.groupby(COL_PRODUCT).agg(
        avg_rating=(COL_RATING, "mean"),
        n_reviews=(COL_RATING, "count"),
        avg_price=(COL_PRICE, "mean"),
    ).reset_index()

    group["score"] = group["avg_rating"] * np.log1p(group["n_reviews"])
    return group


@st.cache_data
def add_sentiment_label(df_source: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df_source[COL_RATING] >= 4,
        df_source[COL_RATING] <= 2,
    ]
    choices = ["positive", "negative"]

    df_local = df_source.copy()
    df_local["sentiment"] = np.select(conditions, choices, default="neutral")
    return df_local


df = add_sentiment_label(df)
df_enriched = enrich_with_country_and_flags(df)
product_stats: pd.DataFrame = compute_product_stats(df_enriched)


# ===================================================
# 4. IA : MODELE DE SENTIMENT
# ===================================================

@st.cache_resource
def load_multilingual_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True
    )

hf_sentiment_model = load_multilingual_sentiment_model()


def predict_sentiment(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"sentiment": "N/A", "confidence": 0.0, "stars": None}

    result = hf_sentiment_model(text)[0]
    label = result["label"]  # ex: "4 stars"
    score = result["score"]

    stars = int(label[0])  # "4 stars" → 4

    if stars <= 2:
        sentiment = "negative"
    elif stars == 3:
        sentiment = "neutral"
    else:
        sentiment = "positive"

    return {
        "sentiment": sentiment,
        "confidence": score,
        "stars": stars,
    }
    
@st.cache_data(show_spinner=False)
def cached_predict_sentiment(text: str) -> str:
    return predict_sentiment(text)["sentiment"]


# ===================================================
# 5. METRIQUES GLOBALES & COUNTRY ANALYTICS
# ===================================================

def compute_overall_metrics_from_enriched(df_source: pd.DataFrame):
    sales_counts = df_source.groupby(COL_PRODUCT).size().sort_values(ascending=False)
    positive_counts = (
        df_source.groupby(COL_PRODUCT)["positive_review"]
        .sum()
        .sort_values(ascending=False)
    )
    negative_counts = (
        df_source.groupby(COL_PRODUCT)["negative_review"]
        .sum()
        .sort_values(ascending=False)
    )
    avg_rating = df_source.groupby(COL_PRODUCT)[COL_RATING].mean().round(2)

    if sales_counts.empty:
        return {}, pd.DataFrame()

    summary = {
        "best_sales": {
            "product": sales_counts.index[0],
            "count": int(sales_counts.iloc[0]),
        },
        "best_positive": {
            "product": positive_counts.index[0],
            "count": int(positive_counts.iloc[0]),
        },
        "worst_sales": {
            "product": sales_counts.index[-1],
            "count": int(sales_counts.iloc[-1]),
        },
        "worst_negative": {
            "product": negative_counts.index[0],
            "count": int(negative_counts.iloc[0]),
        },
        "avg_rating": avg_rating.to_dict(),
    }

    sales_df = sales_counts.reset_index(name="reviews")
    return summary, sales_df


def compute_country_rankings(df_source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, grp in df_source.groupby("country"):
        if grp.empty:
            continue
        stats = grp.groupby(COL_PRODUCT).agg(
            reviews=(COL_PRODUCT, "size"),
            positives=("positive_review", "sum"),
            negatives=("negative_review", "sum"),
            avg_rating=(COL_RATING, "mean"),
        )
        stats["positive_pct"] = (
            stats["positives"] / stats["reviews"].clip(lower=1)
        ) * 100
        stats["negative_pct"] = (
            stats["negatives"] / stats["reviews"].clip(lower=1)
        ) * 100

        best_idx = stats.sort_values(
            by=["positive_pct", "avg_rating", "reviews"],
            ascending=[False, False, False],
        ).index[0]
        worst_idx = stats.sort_values(
            by=["negative_pct", "reviews"],
            ascending=[False, False],
        ).index[0]

        rows.append(
            {
                "country": country,
                "best_product": best_idx,
                "best_positive_pct": round(stats.loc[best_idx, "positive_pct"], 1),
                "worst_product": worst_idx,
                "worst_negative_pct": round(stats.loc[worst_idx, "negative_pct"], 1),
                "samples": len(grp),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("samples", ascending=False)


def build_country_visual_frames(df_source: pd.DataFrame):
    rating_hist = (
        df_source[COL_RATING].value_counts().sort_index().reset_index()
    )
    rating_hist.columns = ["rating", "count"]
    country_counts = df_source.groupby("country").size().reset_index(
        name="review_count"
    )
    return rating_hist, country_counts


def build_product_volume_sentiment(df_source: pd.DataFrame) -> pd.DataFrame:
    if "positive_review" not in df_source.columns or "negative_review" not in df_source.columns:
        tmp = df_source.copy()
        tmp["positive_review"] = tmp[COL_RATING].astype(float) >= 4
        tmp["negative_review"] = tmp[COL_RATING].astype(float) <= 2
    else:
        tmp = df_source

    agg = tmp.groupby(COL_PRODUCT).agg(
        reviews=(COL_RATING, "count"),
        positives=("positive_review", "sum"),
        negatives=("negative_review", "sum"),
    ).reset_index()

    return agg


# ===================================================
# 6. GRAPHIQUES GLOBAUX PRODUITS
# ===================================================

def get_rating_distribution(source_df: Optional[pd.DataFrame] = None):
    data = source_df if source_df is not None else df_enriched
    if COL_RATING not in data.columns:
        return None

    hist = data[COL_RATING].value_counts().sort_index().reset_index()
    hist.columns = ["rating", "count"]

    fig = px.bar(
        hist,
        x="rating",
        y="count",
        title="Distribution globale des notes",
    )
    fig.update_layout(xaxis_title="Note", yaxis_title="Nombre de reviews")
    return fig


def get_price_rating_scatter(stats_df: Optional[pd.DataFrame] = None):
    if COL_PRICE not in df_enriched.columns:
        return None

    stats = stats_df if stats_df is not None else product_stats
    stats = stats[stats["avg_price"].notna()]

    if stats.empty:
        return None

    fig = px.scatter(
        stats,
        x="avg_price",
        y="avg_rating",
        size="n_reviews",
        hover_name=COL_PRODUCT,
        title="Prix moyen vs rating moyen (taille = nb de reviews)",
    )
    fig.update_layout(xaxis_title="Prix moyen", yaxis_title="Rating moyen")
    return fig


def get_price_boxplot(
    selected_products: Optional[List[str]] = None,
    source_df: Optional[pd.DataFrame] = None,
):
    data = source_df.copy() if source_df is not None else df_enriched.copy()
    if COL_PRICE not in data.columns:
        return None

    data = data[data[COL_PRICE].notna()]

    if selected_products:
        data = data[data[COL_PRODUCT].isin(selected_products)]

    if data.empty:
        return None

    fig = px.box(
        data,
        x=COL_PRODUCT,
        y=COL_PRICE,
        title="Distribution des prix par produit",
    )
    fig.update_layout(xaxis_title="Produit", yaxis_title="Prix")
    return fig


# ===================================================
# 7. ANALYSE DÉTAILLÉE PAR PRODUIT
# ===================================================

def get_product_insights(product_name: str, max_reviews: int = 5):
    sub = df_enriched[df_enriched[COL_PRODUCT] == product_name].copy()
    if sub.empty:
        return (
            f"Aucune donnée pour le produit : {product_name}",
            None,
            None,
            pd.DataFrame(columns=[COL_REVIEW_TEXT]),
            "Aucune review disponible pour ce produit.",
        )

    # =========================
    # STATISTIQUES CLASSIQUES
    # =========================
    avg_rating = sub[COL_RATING].mean()
    n_reviews = sub[COL_RATING].count()
    avg_price = sub[COL_PRICE].mean() if COL_PRICE in sub.columns else np.nan
    avg_price_str = f"{avg_price:.2f}" if not np.isnan(avg_price) else "N/A"

    metrics_md = f"""
### 📊 Statistiques pour **{product_name}**

- Rating moyen : **{avg_rating:.2f} / 5**
- Nombre de reviews : **{n_reviews}**
- Prix moyen : **{avg_price_str}**
"""

    # =========================
    # DISTRIBUTION DES NOTES
    # =========================
    rating_counts = sub[COL_RATING].value_counts().sort_index()
    if not rating_counts.empty:
        rating_df = rating_counts.reset_index()
        rating_df.columns = ["rating", "count"]
        fig_ratings = px.bar(
            rating_df,
            x="rating",
            y="count",
            title="Distribution des notes pour ce produit",
        )
        fig_ratings.update_layout(
            xaxis_title="Note",
            yaxis_title="Nombre de reviews",
        )
    else:
        fig_ratings = None

    # =========================
    # ANALYSE DE SENTIMENT (HF) — OPTIMISÉE
    # =========================
    MAX_REVIEWS_FOR_AI = 25  # ← clé performance

    sub_text = (
        sub[sub[COL_REVIEW_TEXT].notna()]
        .head(MAX_REVIEWS_FOR_AI)
        .copy()
    )

    if not sub_text.empty:
        sub_text["predicted_sentiment"] = sub_text[COL_REVIEW_TEXT].apply(
            lambda t: cached_predict_sentiment(t)
        )

        sentiment_counts = (
            sub_text["predicted_sentiment"]
            .value_counts(normalize=True)
            * 100
        )
    else:
        sentiment_counts = pd.Series(dtype=float)

    # =========================
    # GRAPHE DES SENTIMENTS
    # =========================
    if not sentiment_counts.empty:
        sentiment_df = sentiment_counts.reset_index()
        sentiment_df.columns = ["sentiment", "percentage"]

        fig_sentiment = px.bar(
            sentiment_df,
            x="sentiment",
            y="percentage",
            title="Répartition des sentiments prédits pour ce produit",
            text="percentage",
        )
        fig_sentiment.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="% des reviews",
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )
    else:
        fig_sentiment = None

    # =========================
    # REVIEWS EXEMPLES
    # =========================
    sample_reviews = sub_text[[COL_REVIEW_TEXT]].head(max_reviews)

    # =========================
    # SYNTHÈSE IA
    # =========================
    if not sentiment_counts.empty:
        pos = sentiment_counts.get("positive", 0.0)
        neg = sentiment_counts.get("negative", 0.0)
        neu = sentiment_counts.get("neutral", 0.0)

        summary = (
            f"- **{pos:.1f}%** des reviews sont **positives** 😄\n"
            f"- **{neg:.1f}%** sont **négatives** 😡\n"
            f"- **{neu:.1f}%** sont **neutres** 😐\n"
        )
    else:
        summary = "Pas assez de texte pour analyser le sentiment de ce produit."

    summary_md = f"### 🧠 Synthèse IA des avis\n\n{summary}"

    return metrics_md, fig_ratings, fig_sentiment, sample_reviews, summary_md


# ===================================================
# 8. AIDE À LA DÉCISION & RECOMMANDATIONS
# ===================================================

def evaluate_product(product_name: str) -> Dict[str, Any]:
    sub = df_enriched[df_enriched[COL_PRODUCT] == product_name].copy()
    if sub.empty:
        return {"verdict": "Inconnu", "details": "Aucune donnée pour ce produit."}

    avg_rating = sub[COL_RATING].mean()
    n_reviews = sub[COL_RATING].count()

    # =========================
    # ANALYSE DE SENTIMENT (HF) — OPTIMISÉE
    # =========================
    MAX_REVIEWS_FOR_AI = 25  # ← même règle partout

    sub_text = (
        sub[sub[COL_REVIEW_TEXT].notna()]
        .head(MAX_REVIEWS_FOR_AI)
        .copy()
    )

    if not sub_text.empty:
        sub_text["predicted_sentiment"] = sub_text[COL_REVIEW_TEXT].apply(
            lambda t: cached_predict_sentiment(t)
        )

        sentiment_counts = (
            sub_text["predicted_sentiment"]
            .value_counts(normalize=True)
            * 100
        )
    else:
        sentiment_counts = pd.Series(dtype=float)

    pos = sentiment_counts.get("positive", 0.0)
    neg = sentiment_counts.get("negative", 0.0)

    # =========================
    # RÈGLES MÉTIER (INCHANGÉES)
    # =========================
    if (avg_rating >= 4.2) and (n_reviews >= 30) and (pos >= 65) and (neg <= 15):
        verdict = "✅ Très bon produit (fortement recommandé)"
    elif (avg_rating >= 3.5) and (pos >= 45) and (neg <= 30):
        verdict = "🟡 Produit correct / acceptable"
    else:
        verdict = "⚠️ Produit à risque (beaucoup d’insatisfaction potentielle)"

    details = (
        f"- Rating moyen : {avg_rating:.2f} / 5\n"
        f"- Nombre de reviews : {n_reviews}\n"
        f"- % reviews positives (IA) : {pos:.1f}%\n"
        f"- % reviews négatives (IA) : {neg:.1f}%\n"
    )

    return {"verdict": verdict, "details": details}



@st.cache_resource
def build_similarity_engine(df_source: pd.DataFrame):
    agg = (
        df_source.groupby(COL_PRODUCT)[COL_REVIEW_TEXT]
        .apply(lambda s: " ".join(s.astype(str).tolist()))
        .reset_index()
    )

    agg["agg_text"] = agg[COL_REVIEW_TEXT].fillna("")
    # 🔹 Pas de stop_words="english" → multilingue OK
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(agg["agg_text"])

    sim_matrix = cosine_similarity(X)
    products = agg[COL_PRODUCT].tolist()

    return products, sim_matrix


similarity_products, similarity_matrix = build_similarity_engine(df_enriched)


def get_similar_products(product_name: str, top_k: int = 3) -> pd.DataFrame:
    if product_name not in similarity_products:
        return pd.DataFrame(columns=["product", "similarity"])

    idx = similarity_products.index(product_name)
    sims = similarity_matrix[idx]

    indices = np.argsort(sims)[::-1]
    similar_items = []
    for i in indices:
        if i == idx:
            continue
        similar_items.append((similarity_products[i], sims[i]))
        if len(similar_items) >= top_k:
            break

    return pd.DataFrame(similar_items, columns=["product", "similarity"])


def compare_products(product_list: List[str]) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    if not product_list:
        return pd.DataFrame(), None, None

    sub_stats = product_stats[product_stats[COL_PRODUCT].isin(product_list)].copy()
    if sub_stats.empty:
        return pd.DataFrame(), None, None

    sub_stats = sub_stats.sort_values(by="score", ascending=False)
    best_row = sub_stats.iloc[0]
    best_product = best_row[COL_PRODUCT]

    avg_rating_best = best_row["avg_rating"]
    n_reviews_best = int(best_row["n_reviews"])
    avg_price_best = best_row["avg_price"]
    score_best = best_row["score"]

    if len(sub_stats) > 1:
        others = sub_stats.iloc[1:]
        mean_rating_others = others["avg_rating"].mean()
        mean_n_reviews_others = others["n_reviews"].mean()
        mean_price_others = others["avg_price"].mean()

        delta_price = avg_price_best - mean_price_others

        justification = f"""
### Pourquoi ce produit est recommandé ?

- **Note moyenne plus élevée** : {avg_rating_best:.2f} vs {mean_rating_others:.2f} en moyenne pour les autres produits.
- **Plus de retours clients** : {n_reviews_best} reviews vs {mean_n_reviews_others:.0f} en moyenne, ce qui rend la note plus fiable.
- **Score global le plus élevé** (rating × log(1 + nb reviews)) : {score_best:.2f}, supérieur aux autres produits.
{"- 💰 Son prix est **un peu plus élevé** que la moyenne, ce qui peut se justifier par une meilleure qualité perçue."
 if delta_price > 0 else "- 💰 Son prix est **inférieur ou similaire** à la moyenne, ce qui améliore son rapport qualité/prix."}
"""
    else:
        justification = f"""
### Pourquoi ce produit est recommandé ?

- C'est le seul produit sélectionné.
- Note moyenne : **{avg_rating_best:.2f} / 5** avec **{n_reviews_best}** reviews.
- Score global : **{score_best:.2f}**.
"""

    return sub_stats, best_product, justification


# ===================================================
# 9. UI STREAMLIT
# ===================================================

st.title("📈 Thorfin Product Insights – Dashboard IA")
st.write(
    "Application interactive pour aider l’équipe produit à analyser les performances des produits "
    "électroniques et électroménagers, avec une fonctionnalité d’**analyse de sentiment IA** basée sur les avis clients, "
    "et une **analyse avancée par pays / langue**."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue d’ensemble",
        "Analyse par produit",
        "Analyse pays & langues",
        "Recommandations & décisions",
        "Démo IA (sentiment)",
        "À propos du modèle IA",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres globaux")

df_filtered = df_enriched.copy()
if COL_RATING in df_filtered.columns:
    min_rating, max_rating = st.sidebar.slider(
        "Filtrer par note",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.5,
    )
    df_filtered = df_filtered[
        (df_filtered[COL_RATING] >= min_rating)
        & (df_filtered[COL_RATING] <= max_rating)
    ]
    st.sidebar.caption(f"{len(df_filtered)} reviews après filtrage.")
else:
    min_rating, max_rating = 1.0, 5.0
    st.sidebar.caption(f"{len(df_filtered)} reviews disponibles.")


# ---------------------------------------------------
# PAGE 1 : Vue d’ensemble
# ---------------------------------------------------
if page == "Vue d’ensemble":
    st.subheader("📌 Vue globale")

    stats_filtered = compute_product_stats(df_filtered)
    summary_global, _ = compute_overall_metrics_from_enriched(df_filtered)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Nb de produits", stats_filtered[COL_PRODUCT].nunique())
    col_m2.metric("Nb de reviews", len(df_filtered))
    col_m3.metric("Note moyenne", f"{df_filtered[COL_RATING].mean():.2f}")
    col_m4.metric(
        "% reviews ≥ 4⭐",
        f"{(df_filtered[COL_RATING] >= 4).mean() * 100:.1f}%"
    )

    st.markdown("---")
    st.subheader("👑 Leaders & Laggards (globaux)")

    if summary_global:

        def leader_card_html(title: str, product: str, subtitle: str, bad: bool = False) -> str:
            extra_class = " bad" if bad else ""
            return f"""
<div class="leader-card{extra_class}">
  <div class="leader-title">{title}</div>
  <div class="leader-product">{product}</div>
  <div class="leader-sub">{subtitle}</div>
</div>
"""

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                leader_card_html(
                    "🏅 Best seller (nb reviews)",
                    summary_global["best_sales"]["product"],
                    f"{summary_global['best_sales']['count']} reviews",
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                leader_card_html(
                    "✅ Best par avis positifs",
                    summary_global["best_positive"]["product"],
                    f"{summary_global['best_positive']['count']} avis positifs",
                ),
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown(
                leader_card_html(
                    "📉 Worst seller",
                    summary_global["worst_sales"]["product"],
                    f"{summary_global['worst_sales']['count']} reviews",
                    bad=True,
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                leader_card_html(
                    "😡 Most negative",
                    summary_global["worst_negative"]["product"],
                    f"{summary_global['worst_negative']['count']} avis négatifs",
                    bad=True,
                ),
                unsafe_allow_html=True,
            )

    else:
        st.info("Pas assez de données pour calculer les leaders / laggards.")

    # -------- Meilleurs & pires produits : tableaux + bulles --------
    st.markdown("---")
    st.subheader("📊 Meilleurs & pires produits (rating moyen vs nombre de reviews)")

    col1, col2 = st.columns(2)
    with col1:
        min_reviews = st.slider(
            "Nombre minimum de reviews par produit",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
        )
    with col2:
        top_n = st.slider(
            "Nombre de produits à afficher (Top / Pires)",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
        )

    stats_for_rank = stats_filtered[stats_filtered["n_reviews"] >= min_reviews].copy()

    if stats_for_rank.empty:
        st.info("Pas assez de données pour ces paramètres.")
    else:
        # Score pondéré (pour la taille des bulles)
        stats_for_rank["weighted_score"] = (
            stats_for_rank["avg_rating"] * np.log1p(stats_for_rank["n_reviews"])
        )

        # Top & bottom
        top_products = stats_for_rank.sort_values(
            by=["avg_rating", "n_reviews"], ascending=[False, False]
        ).head(top_n)

        bottom_products = stats_for_rank.sort_values(
            by=["avg_rating", "n_reviews"], ascending=[True, False]
        ).head(top_n)

        # ---------- Top produits (tableau + graphe côte à côte)
        st.markdown("### ⭐ Top produits")

        col_top_table, col_top_chart = st.columns([1.2, 2])

        with col_top_table:
            st.markdown("**Top produits (triés par rating moyen)**")
            df_top_display = top_products[
                [COL_PRODUCT, "avg_rating", "n_reviews", "weighted_score"]
            ].rename(
                columns={
                    COL_PRODUCT: "product",
                    "avg_rating": "avg_rating",
                    "n_reviews": "num_reviews",
                    "weighted_score": "weighted_score",
                }
            )
            st.dataframe(df_top_display, use_container_width=True)

        with col_top_chart:
            fig_top = px.scatter(
                top_products,
                x="avg_rating",
                y="n_reviews",
                size="weighted_score",
                hover_name=COL_PRODUCT,
                title=f"Top {len(top_products)} produits – rating moyen vs nb de reviews (min {min_reviews} reviews)",
            )
            fig_top.update_layout(
                xaxis_title="Rating moyen",
                yaxis_title="Nombre de reviews",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        # ---------- Pires produits (tableau + graphe côte à côte)
        st.markdown("---")
        st.markdown("### ❌ Pires produits")

        col_bot_table, col_bot_chart = st.columns([1.2, 2])

        with col_bot_table:
            st.markdown("**Pires produits (rating moyen le plus faible)**")
            df_bot_display = bottom_products[
                [COL_PRODUCT, "avg_rating", "n_reviews", "weighted_score"]
            ].rename(
                columns={
                    COL_PRODUCT: "product",
                    "avg_rating": "avg_rating",
                    "n_reviews": "num_reviews",
                    "weighted_score": "weighted_score",
                }
            )
            st.dataframe(df_bot_display, use_container_width=True)

        with col_bot_chart:
            fig_bottom = px.scatter(
                bottom_products,
                x="avg_rating",
                y="n_reviews",
                size="weighted_score",
                hover_name=COL_PRODUCT,
                title=f"Pires {len(bottom_products)} produits – rating moyen vs nb de reviews (min {min_reviews} reviews)",
            )
            fig_bottom.update_layout(
                xaxis_title="Rating moyen",
                yaxis_title="Nombre de reviews",
            )
            st.plotly_chart(fig_bottom, use_container_width=True)

    # -------- Distribution des notes + prix vs rating --------
    st.markdown("---")
    st.subheader("📉 Distribution des notes & relation prix / rating")

    c_dist, c_scatter = st.columns(2)
    with c_dist:
        fig_dist = get_rating_distribution(df_filtered)
        if fig_dist is not None:
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Impossible d'afficher la distribution des notes.")

    with c_scatter:
        fig_scatter = get_price_rating_scatter(stats_filtered)
        if fig_scatter is not None:
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Pas assez de données de prix pour afficher le scatter.")

    # -------- Distribution des prix --------
    st.markdown("---")
    st.subheader("💰 Distribution des prix par produit")

    products_list: List[str] = sorted(df_filtered[COL_PRODUCT].dropna().unique().tolist())
    selected_products = st.multiselect(
        "Filtrer sur certains produits (optionnel)",
        options=products_list,
    )

    fig_price = get_price_boxplot(selected_products or None, source_df=df_filtered)
    if fig_price is not None:
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Pas de données de prix disponibles.")

    # -------- Volume & sentiment par produit --------
    st.markdown("---")
    st.subheader("📦 Volume & sentiment par produit (Top N)")

    prod_vol_df = build_product_volume_sentiment(df_filtered)

    if prod_vol_df.empty:
        st.info("Pas assez de données pour afficher les volumes par produit.")
    else:
        top_k_products = st.slider(
            "Nombre de produits à afficher (Top N par nombre de reviews)",
            min_value=3,
            max_value=min(30, len(prod_vol_df)),
            value=min(10, len(prod_vol_df)),
            step=1,
        )

        prod_top = prod_vol_df.sort_values("reviews", ascending=False).head(top_k_products)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            fig_reviews = px.bar(
                prod_top,
                x=COL_PRODUCT,
                y="reviews",
                title="Nombre de reviews par produit (Top N)",
            )
            fig_reviews.update_layout(
                xaxis_title="Produit",
                yaxis_title="Nombre de reviews",
            )
            st.plotly_chart(fig_reviews, use_container_width=True)

        with col_p2:
            prod_long = prod_top.melt(
                id_vars=[COL_PRODUCT, "reviews"],
                value_vars=["positives", "negatives"],
                var_name="sentiment_type",
                value_name="count",
            )

            label_map = {
                "positives": "Avis positifs",
                "negatives": "Avis négatifs",
            }
            prod_long["sentiment_type"] = prod_long["sentiment_type"].map(label_map)

            fig_sent_prod = px.bar(
                prod_long,
                x=COL_PRODUCT,
                y="count",
                color="sentiment_type",
                barmode="group",
                title="Avis positifs / négatifs par produit (Top N)",
            )
            fig_sent_prod.update_layout(
                xaxis_title="Produit",
                yaxis_title="Nombre de reviews",
                legend_title="Type d'avis",
            )
            st.plotly_chart(fig_sent_prod, use_container_width=True)


# ---------------------------------------------------
# PAGE 2 : Analyse par produit
# ---------------------------------------------------
elif page == "Analyse par produit":
    st.subheader("🔍 Analyse détaillée par produit")

    products_list: List[str] = sorted(df_enriched[COL_PRODUCT].dropna().unique().tolist())

    product_name = st.selectbox("Choisir un produit", options=products_list)
    n_reviews_display = st.slider(
        "Nombre de reviews à afficher",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    (
        metrics_md,
        fig_ratings,
        fig_sentiment,
        sample_reviews,
        summary_md,
    ) = get_product_insights(product_name, max_reviews=n_reviews_display)

    st.markdown(metrics_md)

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        if fig_ratings is not None:
            st.plotly_chart(fig_ratings, use_container_width=True)
        else:
            st.info("Pas assez de reviews pour afficher la distribution des notes.")
    with col_plot2:
        if fig_sentiment is not None:
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("Pas assez de reviews pour afficher la répartition des sentiments.")

    st.markdown(summary_md)

    st.markdown("### 📝 Exemples de reviews")
    st.dataframe(sample_reviews, use_container_width=True)


# ---------------------------------------------------
# PAGE 3 : Analyse pays & langues
# ---------------------------------------------------
elif page == "Analyse pays & langues":
    st.subheader("🌍 Analyse par pays & langues")

    if COL_LANGUAGE not in df_enriched.columns:
        st.info(
            "La colonne de langue n'est pas disponible dans le CSV "
            f"(`{COL_LANGUAGE}`). L'analyse par pays utilise alors la colonne 'country' "
            "ou affiche 'Unknown'."
        )

    rating_hist_country, country_counts = build_country_visual_frames(df_filtered)
    country_df = compute_country_rankings(df_filtered)

    st.markdown("### 📋 Classement par pays (meilleur / pire produit)")

    if country_df.empty:
        st.info(
            "Données pays indisponibles ou insuffisantes. "
            "Vérifie que le CSV contient bien les colonnes de langue / pays."
        )
    else:
        st.dataframe(country_df, use_container_width=True)

    st.markdown("---")
    col_bar, col_map = st.columns(2)

    with col_bar:
        st.markdown("### 📦 Volume de reviews par pays")
        if country_counts.empty:
            st.info("Aucun pays détecté dans les données.")
        else:
            fig_country_bar = px.bar(
                country_counts,
                x="country",
                y="review_count",
                title="Nombre de reviews par pays",
            )
            st.plotly_chart(fig_country_bar, use_container_width=True)

    with col_map:
        st.markdown("### 🗺️ Carte des reviews par pays")
        if country_counts.empty:
            st.info("Pas assez de données pour afficher la carte.")
        else:
            fig_map = px.choropleth(
                country_counts,
                locations="country",
                locationmode="country names",
                color="review_count",
                title="Reviews par pays",
            )
            st.plotly_chart(fig_map, use_container_width=True)

    # 🔥 Sentiment par pays avec positifs / négatifs / neutres
    st.markdown("---")
    st.markdown("### 😀😐😡 Sentiment par pays")

    if df_filtered["country"].nunique() <= 1:
        st.info("Pas assez de diversité de pays pour afficher le sentiment par pays.")
    else:
        country_sent = df_filtered.groupby("country").agg(
            reviews=("country", "size"),
            positives=("positive_review", "sum"),
            negatives=("negative_review", "sum"),
        ).reset_index()

        # On déduit les neutres
        country_sent["neutrals"] = (
            country_sent["reviews"]
            - country_sent["positives"]
            - country_sent["negatives"]
        )

        country_sent["positive_pct"] = (
            country_sent["positives"] / country_sent["reviews"].clip(lower=1)
        ) * 100
        country_sent["negative_pct"] = (
            country_sent["negatives"] / country_sent["reviews"].clip(lower=1)
        ) * 100
        country_sent["neutral_pct"] = (
            country_sent["neutrals"] / country_sent["reviews"].clip(lower=1)
        ) * 100

        sent_long = country_sent.melt(
            id_vars=["country", "reviews"],
            value_vars=["positive_pct", "negative_pct", "neutral_pct"],
            var_name="sentiment_type",
            value_name="pct",
        )

        label_map_country = {
            "positive_pct": "% positifs",
            "negative_pct": "% négatifs",
            "neutral_pct": "% neutres",
        }
        sent_long["sentiment_type"] = sent_long["sentiment_type"].map(label_map_country)

        fig_sent_country = px.bar(
            sent_long,
            x="country",
            y="pct",
            color="sentiment_type",
            barmode="group",
            title="% d'avis positifs / négatifs / neutres par pays",
        )
        fig_sent_country.update_layout(
            xaxis_title="Pays",
            yaxis_title="% des reviews",
            legend_title="Type de sentiment",
        )
        st.plotly_chart(fig_sent_country, use_container_width=True)


# ---------------------------------------------------
# PAGE 4 : Recommandations & décisions
# ---------------------------------------------------
elif page == "Recommandations & décisions":
    st.subheader("🎯 Aide à la décision & recommandations produits")

    products_list: List[str] = sorted(df_enriched[COL_PRODUCT].dropna().unique().tolist())

    st.markdown("#### 1️⃣ Comparer plusieurs produits disponibles")

    selected_compare = st.multiselect(
        "Choisis les produits que tu proposerais à un client :",
        options=products_list,
    )

    if st.button("Analyser et recommander parmi ces produits"):
        table_compare, best_prod, justification = compare_products(selected_compare)
        if table_compare.empty:
            st.info("Aucune donnée pour ces produits.")
        else:
            st.markdown("##### Résumé des produits sélectionnés (triés par score décroissant)")
            st.dataframe(
                table_compare[[COL_PRODUCT, "avg_rating", "n_reviews", "avg_price", "score"]],
                use_container_width=True,
            )

            fig_scores = px.bar(
                table_compare,
                x=COL_PRODUCT,
                y="score",
                title="Score global des produits sélectionnés",
            )
            st.plotly_chart(fig_scores, use_container_width=True)

            if best_prod:
                st.success(f"👉 Produit recommandé en priorité : **{best_prod}**")
            if justification:
                st.markdown(justification)

    st.markdown("---")
    st.markdown("#### 2️⃣ Est-ce que ce produit est bien ou pas ?")

    prod_to_eval = st.selectbox(
        "Sélectionne un produit à évaluer",
        options=products_list,
        key="prod_eval",
    )

    if prod_to_eval:
        eval_res = evaluate_product(prod_to_eval)
        st.markdown(eval_res["verdict"])
        st.markdown(eval_res["details"])

    st.markdown("---")
    st.markdown("#### 3️⃣ Produits similaires (pour proposer des alternatives au client)")

    prod_for_sim = st.selectbox(
        "Choisis un produit pour voir des alternatives similaires",
        options=products_list,
        key="prod_sim",
    )

    if prod_for_sim:
        df_sim = get_similar_products(prod_for_sim, top_k=3)
        if df_sim.empty:
            st.info("Pas assez de texte pour trouver des produits similaires.")
        else:
            st.markdown(f"Produits similaires à **{prod_for_sim}** :")
            df_sim["similarity"] = (df_sim["similarity"] * 100).round(1)
            df_sim = df_sim.rename(columns={"product": "Produit", "similarity": "Similarité (%)"})
            st.dataframe(df_sim, use_container_width=True)


# ---------------------------------------------------
# PAGE 5 : Démo IA
# ---------------------------------------------------
elif page == "Démo IA (sentiment)":
    st.subheader("🧠 Démo d’analyse de sentiment sur un avis texte")

    text_input = st.text_area(
        "Écris un avis client :",
        placeholder="Exemple : This product is amazing, I use it every day...",
        height=150,
    )

    if st.button("Prédire le sentiment"):
        res = predict_sentiment(text_input)

        if res["sentiment"] == "N/A":
            st.warning("Veuillez saisir un texte.")
        else:
            sentiment = res["sentiment"]
            confidence = res["confidence"]
            stars = res["stars"]

            if sentiment == "positive":
                st.success(f"Sentiment prédit : **positive** 😄 ({stars}⭐ – confiance {confidence:.0%})")
            elif sentiment == "negative":
                st.error(f"Sentiment prédit : **negative** 😡 ({stars}⭐ – confiance {confidence:.0%})")
            else:
                st.info(f"Sentiment prédit : **neutral** 😐 ({stars}⭐ – confiance {confidence:.0%})")
                

# ---------------------------------------------------
# PAGE 6 : À propos du modèle IA
# ---------------------------------------------------
elif page == "À propos du modèle IA":
    st.subheader("🔎 Détails techniques du modèle de sentiment")

    st.markdown(
        """
### 🤖 Modèle de sentiment utilisé

- **Nom** : `nlptown/bert-base-multilingual-uncased-sentiment`
- **Type** : Transformer (BERT)
- **Framework** : Hugging Face Transformers
- **Langues supportées** : Multilingue (FR, EN, ES, DE, IT, PT, etc.)
- **Sortie du modèle** : Note de **1 à 5 étoiles**
- **Interprétation métier** :
  - ⭐ 1–2 → **Negative**
  - ⭐ 3 → **Neutral**
  - ⭐ 4–5 → **Positive**

### 🎯 Pourquoi ce modèle ?
- Comprend le **contexte et la négation**
- Robuste aux **avis multilingues**
- Beaucoup plus fiable que TF-IDF sur des textes réels
- Utilisé en **production** pour l’analyse d’avis clients

### ⚠️ À propos des métriques
Ce modèle est **pré-entraîné** sur de grands jeux de données publics.
Les métriques d'entraînement (accuracy, recall, etc.) ne sont pas recalculées localement,
car le modèle est utilisé **tel quel (inférence uniquement)**.

👉 La qualité est évaluée via :
- tests manuels multilingues
- cohérence des prédictions
- comparaison avec les notes réelles (ratings)
"""
    )

    st.markdown("---")
    st.markdown("### 📊 Répartition des classes de sentiment (basée sur les ratings réels)")

    # =========================
    # DONNÉES
    # =========================
    class_counts = df_enriched["sentiment"].value_counts().reset_index()
    class_counts.columns = ["sentiment", "count"]

    # =========================
    # GRAPHIQUE CORRIGÉ
    # =========================
    fig_classes = px.bar(
        class_counts,
        x="sentiment",
        y="count",
        color="sentiment",
        text="count",
        title="Nombre de reviews par classe de sentiment (labels dérivés des ratings)",
        color_discrete_map={
            "positive": "#22c55e",  # vert
            "negative": "#ef4444",  # rouge
            "neutral": "#f59e0b",   # orange
        },
    )

    fig_classes.update_traces(
        textposition="inside",
        textfont_size=14,
        textfont_color="white",
    )

    fig_classes.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Nombre de reviews",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    st.plotly_chart(fig_classes, use_container_width=True)
