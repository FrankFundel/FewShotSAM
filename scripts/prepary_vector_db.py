'''
# https://arxiv.org/pdf/2304.07193.pdf
# https://www.facebook.com/groups/faissusers/posts/1025663204524632/

import faiss

# Initialize the Faiss GPU index with inner product (dot product) similarity
index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), extracted_features.shape[1])

# Add the feature embeddings to the GPU index
index.add(extracted_features)

# Use dot product similarity for querying on GPU
query_vector = np.random.rand(1, extracted_features.shape[1])  # Replace with your query vector
k = 10  # Number of nearest neighbors to retrieve
distances, indices = gpu_index.search_and_reconstruct(query_vector, k)

for each image, mask in ds:
    cutouts = create_cutouts(image, mask)
    cutout_embeddings = embed(cutouts)
    ids = index.add(cutout_embeddings)
    
'''