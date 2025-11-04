import faiss
import h5py
import numpy as np
import os
import struct
import requests


# Part - 0: Creating Index and Performing Queries

M = 16 # (number of bi-directional links per node)
EF_CONSTRUCTION = 200 # (size of dynamic candidate list during construction)
EF_SEARCH = 200

def read_fvecs_to_np(file_path):
    vecs = []
    with open(file_path, 'rb') as f:
        while True:
            # Read dimension (4 bytes, little-endian int)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<i', dim_bytes)[0]

            # Read vector values (dim * 4 bytes, little-endian floats)
            vec = struct.unpack('<' + 'f' * dim, f.read(dim * 4))
            vecs.append(vec)

    return np.array(vecs, dtype=np.float32)

def import_sift1(base_path):
    data_path = os.path.join(base_path, 'sift_base.fvecs')
    query_path = os.path.join(base_path, 'sift_query.fvecs')

    database_vecs = read_fvecs_to_np(data_path)
    query_vecs = read_fvecs_to_np(query_path)
    return database_vecs, query_vecs


def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # The SIFT1M dataset use three different file formats .bvecs, .fvecs and .ivecs
    #     • The vector files are stored in .bvecs or .fvecs format,
    #     • The groundtruth file in is .ivecs format.
    # The vectors are stored in raw little endian.
    # Each vector takes 4+d*4 bytes for .fvecs and .ivecs formats,
    # and 4+d bytes for .bvecs formats, where d is the dimensionality of the vector

    base_path = './sift'
    print("(evaluate_hnsw) Loading SIFT1M dataset...")

    database_vectors, query_vectors = import_sift1(base_path)
    print(f"(evaluate_hnsw) Loaded {database_vectors.shape[0]} database vectors of dimension {database_vectors.shape[1]}")
    print(f"(evaluate_hnsw) Loaded {query_vectors.shape[0]} query vectors of dimension {query_vectors.shape[1]}")

    # Get dimensionality
    d = database_vectors.shape[1]  # Should be 128 for SIFT1M

    print(f"(evaluate_hnsw) Building HNSW index... for dimensionality: {d}")

    index = faiss.IndexHNSWFlat(d, M)

    # Set efConstruction parameter
    index.hnsw.efConstruction = EF_CONSTRUCTION

    # Add database vectors to the index
    print("(evaluate_hnsw) Adding vectors to index...")
    index.add(database_vectors)
    print(f"(evaluate_hnsw) Index built with {index.ntotal} vectors")

    # Set efSearch parameter for querying
    index.hnsw.efSearch = EF_SEARCH

    print("Performing query...")

    # Use the first query vector
    query_vector = query_vectors[0:1]  # Shape: (1, 128)

    # Search for top 10 nearest neighbors
    k = 10
    distances, indices = index.search(query_vector, k)

    print(f"(evaluate_hnsw) Top {k} nearest neighbors:")
    print(f"(evaluate_hnsw) Indices: {indices[0]}")
    print(f"(evaluate_hnsw) Distances: {distances[0]}")

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    output_file = './output.txt'
    with open(output_file, 'w') as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")

    print(f"(evaluate_hnsw) Results written to {output_file}")
    

if __name__ == "__main__":
    evaluate_hnsw()
