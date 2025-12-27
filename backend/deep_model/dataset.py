import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SpillDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        for label, folder in enumerate(["fresh", "recent", "old"]):
            path = os.path.join(root, folder)
            for img in os.listdir(path):
                self.samples.append((os.path.join(path, img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # approximate continuous labels
        age_map = {0: 15, 1: 90, 2: 240}
        thickness_map = {0: 0.9, 1: 0.5, 2: 0.2}

        return img, age_map[label], thickness_map[label], label
