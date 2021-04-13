from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
import glob

class Fpidataset(Dataset):
    # Constructor
    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform
        self.classes = ['Shirts', 'Watches', "Tshirts", "Casual Shoes", "Handbags", "Tops", "Kurtas", "Sports Shoes", "Heels", "Sunglasses"]

        df = pd.read_csv('data/styles.csv', error_bad_lines=False)
        df['image_path'] = df.apply(lambda x: os.path.join("data/images", str(x.id) + ".jpg"), axis=1)
        df = df.drop([32309, 40000, 36381, 16194, 6695]) #drop rows with no image

        temp = df.articleType.value_counts().sort_values(ascending=False)[:10].index.tolist()
        df = df[df["articleType"].isin(temp)]

        # map articleType as number
        mapper = {}
        for i, cat in enumerate(list(df.articleType.unique())):
            mapper[cat] = i
        print(mapper)
        df['targets'] = df.articleType.map(mapper)

        if self.train:
            self.df = get_i_items(df,temp,0, 800)
            self.targets =  self.df.targets.tolist()
        else:
            self.df = get_i_items(df,temp,800, 1000)
            self.targets = self.df.targets.tolist()

    # Get the length
    def __len__(self):
        return len(self.df)



    def __getitem__(self, idx):
        # get imagepath
        img_path = self.df.image_path[idx]

        # open as PIL Image
        image = Image.open(img_path).convert('RGB')
        img_size = image.size

        # resize
        image = image.resize((60, 80))

        # transform
        if self.transform is not None:
            image = self.transform(image)

        # get label
        target = self.targets[idx]

        # get classname
        classname = self.classes[target]

        out = {'image': image, 'target': target, 'meta': {'im_size': img_size, 'index': idx, 'class name': classname}}

        return out

        # get i items of each condition
def get_i_items(df, temp, start, stop):

        # generate new empty dataframe with the columns of the original
        dataframe = df[:0]

        # for each targetclass in temp add i items to dataframe
        for label in temp:
            # print("Füge Items mit target", label, "ein.")
            dataframe = dataframe.append(df[df.articleType == label][start:stop])
            # print("Anzahl items in dataframe", len(dataframe))

        dataframe = dataframe.reset_index()

        return dataframe
