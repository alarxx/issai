import os
from PIL import Image
from torch.utils.data import Dataset

# Like ImageFolder
# dataset = datasets.ImageFolder(root, transform)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.classes = []
        self.idx_to_class = {}

        # assign index to each class folder
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path): # is it necessary?
                self.classes.append(class_name)
                self.idx_to_class[idx] = class_name # ???
                for fname in os.listdir(class_path):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, fname))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
