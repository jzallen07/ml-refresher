# Lesson 07: Data Management - Downloading & Custom Datasets

## Overview

Real-world ML requires working with real datasets. This lesson covers how to download data programmatically, build custom `Dataset` classes, and handle image data efficiently. The custom dataset pattern is especially important for image data where we want to load images lazily (on-demand) to avoid memory overflow.

> **Source**: This lesson is based on the "Data Management: datasets, data splitting, and dataloaders" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Download and extract datasets programmatically
2. Build custom `Dataset` classes with `__init__`, `__len__`, `__getitem__`
3. Handle image data with lazy loading
4. Load and display images with PIL and matplotlib

## Key Concepts

### Why Custom Datasets?

As TK explains: "Specifically for image data, this idea of custom datasets is quite interesting because we can 'lazily' open each input data (pixels) as images individually on demand, so we don't overflow the memory usage."

### Downloading Data

```python
import requests
import tarfile

# Download the data
url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
response = requests.get(url, stream=True)

# Save and extract
with open(tgz_path, "wb") as file:
    for data in tqdm(response.iter_content(chunk_size=1024)):
        file.write(data)

with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(root_dir)
```

### The Dataset Class Interface

A custom dataset needs three methods:

```python
class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir):
        # Setup the data and data paths
        ...

    def __len__(self):
        # Return sample size
        return len(self.labels)

    def __getitem__(self, idx):
        # Get item from sample through index
        # Returns the image and the label
        ...
```

### TK's Oxford Flowers Dataset Implementation

```python
class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "jpg")
        labels_mat = scipy.io.loadmat(os.path.join(root_dir, "imagelabels.mat"))
        self.labels = labels_mat["labels"][0] - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = f"image_{idx+1:05d}.jpg"
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path)
        label = self.labels[idx]
        return image, label
```

### Using the Dataset

```python
dataset = OxfordFlowersDataset(root_dir)
img, label = dataset[0]
# img: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=591x500>
# label: 76
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Creating the data directory structure
2. Downloading the Oxford Flowers dataset
3. Extracting the tgz file
4. Downloading the labels file
5. Building the custom dataset class
6. Loading and displaying images

**Note**: This lesson downloads ~330MB of image data to the `data/` directory.

## Practice Problems

After completing the main lesson, try these exercises:

1. **Add Transform Support**: Modify the dataset class to accept an optional `transform` parameter that's applied to images in `__getitem__`.

2. **Implement `__repr__`**: Add a `__repr__` method that returns useful information like dataset size and image directory path.

3. **Subset Dataset**: Create a version of the dataset that only includes certain flower classes (e.g., classes 0-9 for a simpler classification task).

## Key Takeaways

- Use `requests` with streaming for large file downloads
- `tarfile` handles `.tgz` extraction
- Custom datasets inherit from `torch.utils.data.Dataset`
- Implement `__init__`, `__len__`, and `__getitem__`
- Lazy loading (opening images in `__getitem__`) saves memory
- PIL's `Image.open()` loads images that can be converted to tensors

## Next Lesson

[Lesson 08: Data Management - Transforms & Augmentation](../08_data_transforms/README.md) - Learn to preprocess and augment images for training.
