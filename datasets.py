from torch.utils.data import Dataset
from PIL import Image
import os


class BatchDataset(Dataset):

    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = os.listdir(image_dir)

    def __getitem__(self, index):
        class_ = ['cloudy', 'dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'shine', 'snow', 'sunrise']
        filename = self.filenames[index]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        weather = filename.split(".")[0].split("-")[0]
        label = class_.index(weather)

        return image, label, filename

    def __len__(self):
        return len(self.filenames)
