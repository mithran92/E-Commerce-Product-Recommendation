
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Load the data with error handling
# -------------------------------
if not os.path.exists("clean_data.csv") or not os.path.exists("trending_products.csv"):
    st.error("Please run the notebook to generate 'clean_data.csv' and 'trending_products.csv' first.")
    st.stop()

df = pd.read_csv("clean_data.csv")
trending_df = pd.read_csv("trending_products.csv")

# Ensure required columns exist
required_cols = ["Tags", "Name", "ImageURL", "Rating", "Brand", "ReviewCount"]
for col in required_cols:
    if col not in df.columns:
        df[col] = ""
    if col not in trending_df.columns:
        trending_df[col] = ""

# Fill NA just in case
df["Tags"] = df["Tags"].fillna("")
df["Name"] = df["Name"].fillna("Unnamed Product")
df["ImageURL"] = df["ImageURL"].fillna("")
df["Rating"] = df["Rating"].fillna(0)
df["Brand"] = df["Brand"].fillna("")
df["ReviewCount"] = df["ReviewCount"].fillna(0)
trending_df["ImageURL"] = trending_df["ImageURL"].fillna("")
trending_df["Rating"] = trending_df["Rating"].fillna(0)
trending_df["Brand"] = trending_df["Brand"].fillna("")
trending_df["ReviewCount"] = trending_df["ReviewCount"].fillna(0)


# -------------------------------
# Vectorize the tags for recommendation
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["Tags"].astype(str))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Product indices lookup (case-insensitive)
indices = pd.Series(df.index, index=df["Name"].str.lower()).drop_duplicates()


# -------------------------------
# Recommendation function (case-insensitive)
# -------------------------------
def get_recommendations(product_name, top_n=5):
    name_key = product_name.lower()
    if name_key not in indices:
        return pd.DataFrame()
    idx = indices[name_key]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="E-Commerce Recommender", layout="wide")

st.title("🛍️ E-Commerce Product Recommender")

# Show trending products
st.subheader("🔥 Trending Products")


PLACEHOLDER_IMG = "https://via.placeholder.com/100?text=No+Image"


def get_valid_image_url(url):
    if pd.isna(url):
        return PLACEHOLDER_IMG
    if not isinstance(url, str):
        return PLACEHOLDER_IMG
    url = url.strip()
    if not url or url.lower() in ['nan', 'none', 'null', '']:
        return PLACEHOLDER_IMG
    if not (url.lower().startswith('http://') or url.lower().startswith('https://')):
        return PLACEHOLDER_IMG
    # Optionally, check for common image extensions
    valid_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    if not any(url.lower().endswith(ext) for ext in valid_exts):
        # Allow if it's a valid http(s) url, even if no extension
        return url
    return url

for _, row in trending_df.iterrows():
    col1, col2 = st.columns([1, 4])
    with col1:
        img_url = get_valid_image_url(row["ImageURL"])
        st.image(img_url, width=100)
    with col2:
        st.markdown(f"**{row['Name']}**")
        st.markdown(f"⭐ {row['Rating']} | 🏷️ {row['Brand']} | 💬 Reviews: {row['ReviewCount']}")
    st.markdown("---")


# Product recommendation section
st.subheader("🎯 Get Product Recommendations")

product_input = st.selectbox("Choose a product:", options=df["Name"].unique())
top_n = st.slider("Number of recommendations", 1, 10, 5)

results = get_recommendations(product_input, top_n)

if results.empty:
    st.warning("⚠️ No similar products found.")
else:
    st.success(f"Recommendations for: **{product_input}**")
    for _, row in results.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            img_url = get_valid_image_url(row["ImageURL"])
            st.image(img_url, width=100)
        with col2:
            st.markdown(f"**{row['Name']}**")
            st.markdown(f"⭐ {row['Rating']} | 🏷️ {row['Brand']} | 💬 Reviews: {row['ReviewCount']}")
        st.markdown("---")
