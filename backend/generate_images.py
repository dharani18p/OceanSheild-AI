import cv2
import numpy as np
import os
import random

BASE_DIR = "data"
classes = {
    "fresh": (200, 255),    # bright, strong colors
    "recent": (120, 200),   # medium intensity
    "old": (40, 120)        # faded
}

os.makedirs(BASE_DIR, exist_ok=True)

for cls, intensity_range in classes.items():
    folder = os.path.join(BASE_DIR, cls)
    os.makedirs(folder, exist_ok=True)

    for i in range(10):  # generate 10 images per class
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        # simulate oil spill blobs
        for _ in range(random.randint(3, 8)):
            center = (random.randint(50, 200), random.randint(50, 200))
            radius = random.randint(20, 60)
            color_val = random.randint(*intensity_range)
            color = (
                color_val,
                random.randint(0, color_val),
                random.randint(0, color_val)
            )
            cv2.circle(img, center, radius, color, -1)

        # add blur to simulate aging
        if cls == "old":
            img = cv2.GaussianBlur(img, (31, 31), 0)
        elif cls == "recent":
            img = cv2.GaussianBlur(img, (15, 15), 0)

        cv2.imwrite(os.path.join(folder, f"{cls}_{i}.jpg"), img)

print("✅ Synthetic oil spill images generated successfully")
