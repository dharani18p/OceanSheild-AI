import cv2
import numpy as np

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Color feature
    mean_color = np.mean(img)

    # Edge feature
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size

    # Texture feature
    texture = np.var(gray)

    return [mean_color, edge_density, texture]
