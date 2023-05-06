import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

# Load data
csv_file_path = 'dataset.csv'
df = pd.read_csv(csv_file_path)

# Calculate or load embeddings
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
descriptions = df['PROD_DESCRIPTION'].tolist()
embeddings_file = 'description_embeddings.npy'

if not os.path.exists(embeddings_file):
    description_embeddings = model.encode(descriptions)
    np.save(embeddings_file, description_embeddings)
else:
    description_embeddings = np.load(embeddings_file)

# Streamlit app
st.title("Product Search 🦜")

query = st.text_input("Enter your query:")

if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, description_embeddings)
    top_n = 5
    top_n_indices = similarities[0].argsort()[-top_n:][::-1]

    if len(top_n_indices) == 0:
        st.write("NO PRODUCTS FOUND")
    else:
        st.subheader("Top Matching Products")
        for index in top_n_indices:
            prod_category = df.iloc[index]['PROD_CATEGORY']
            prod_name = df.iloc[index]['PROD_NAME']
            prod_brand = df.iloc[index]['PROD_BRAND']
            prod_price = df.iloc[index]['PROD_PRICE']
            prod_desc = descriptions[index]
            prod_image_url = df.iloc[index]['PROD_IMAGE_URL']
            prod_link = df.iloc[index]['PROD_LINK']

            st.image(prod_image_url, caption="Product Image", width=100)
            st.markdown(f"**Category:** {prod_category}")
            st.markdown(f"**Product Brand:** {prod_brand}")
            st.markdown(f"**Product Name:** {prod_name}")
            st.markdown(f"**Price:** {prod_price}")
            st.markdown(f"**Product Page:** [Link]({prod_link}){{:target='_blank'}}", unsafe_allow_html=True)
            st.markdown(f"**Description:** {prod_desc}")
            st.write("---")


# streamlit run app_embeddings.py

