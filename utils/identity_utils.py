import os
import numpy as np
from utils.face_embedding import get_face_embedding

def predict_identity_from_npy(model, face_img, embeddings_dir="data/embeddings", threshold=0.5):
    """
    So sánh embedding ảnh với tất cả các vector .npy trong thư mục embeddings.
    """
    embedding = get_face_embedding(model, face_img)
    if embedding is None:
        return "Unknown"

    min_dist = float('inf')
    identity = "Unknown"

    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            person = file.split("_")[0]  # Lấy tên người từ file name
            vec_path = os.path.join(embeddings_dir, file)
            vec = np.load(vec_path)

            dist = np.linalg.norm(embedding - vec)
            if dist < min_dist:
                min_dist = dist
                identity = person

    return identity if min_dist < threshold else "Unknown"
