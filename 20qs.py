from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='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')
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token='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')

import json

# Load the nouns
with open('nouns-list.json', 'r') as file:
    data = json.load(file)

# Convert nouns to lowercase if not already
nouns = data['words']

import torch

def get_embeddings(texts):
    # Encode the texts
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    # Get output from model
    with torch.no_grad():
        outputs = model(**encoded_input, output_hidden_states=True)
    # Retrieve the embeddings from the last hidden state
    embeddings = outputs.hidden_states[-1][:, 0, :].detach()
    return embeddings

# Get embeddings for the nouns
noun_embeddings = get_embeddings(nouns)



import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'noun_embeddings' is a Tensor from previous steps

# Convert to NumPy array for compatibility with scikit-learn
noun_embeddings_np = noun_embeddings.numpy()

# Clustering
kmeans = KMeans(n_clusters=10, random_state=0).fit(noun_embeddings_np)
clusters = kmeans.labels_

# Dimensionality Reduction with PCA
pca = PCA(n_components=2)
reduced_pca = pca.fit_transform(noun_embeddings_np)

# Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2, learning_rate='auto', init='random')
reduced_tsne = tsne.fit_transform(noun_embeddings_np)

# Cosine Similarity
similarity_matrix = cosine_similarity(noun_embeddings_np)

# Plotting PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_pca[:, 0], y=reduced_pca[:, 1], hue=clusters, palette='viridis')
plt.title('PCA of Noun Embeddings')
plt.show()

# Plotting t-SNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_tsne[:, 0], y=reduced_tsne[:, 1], hue=clusters, palette='viridis')
plt.title('t-SNE of Noun Embeddings')
plt.show()

# Example of displaying similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Cosine Similarity Matrix of Noun Embeddings')
plt.show()
