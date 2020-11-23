import torch
from torch.utils import data


class TastyVideoDataset(data.Dataset):
    '''
    @param root (str): Path to root directory for Tasty Video Dataset
    @param split (str): Can be 'train' or 'test'
    '''
    def __init__(self, root, split="train", img_transform=None, label_transform=None,
                 test=False):
        assert split in ["train", "val"]
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform

        imgsets_dir = os.path.join(self.root, 'SPLITS/split_2511', '{}ing_set.txt'.format(split))
        with open(imgsets_dir) as imgset_file:
            names = imgset_file.readlines()
        # Remove \n from video names
        names = [name[:-1] for name in names]
        for name in names:
            vid_folder = os.path.join(self.root, 'ALL_RECIPES_without_videos', name)
            frames = os.listdir(os.path.join(vid_folder, 'frames'))
            with open(os.path.join(vid_folder, 'csvalignment.dat'), 'r') as alignmentFile:
                alignment_file = alignmentFile.readlines()
            self.files[split].append({
                    "frames": frames,
                    "alignment": alignment_file
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        '''
        img_file = datafiles["img"]
        domain = datafiles["domain_label"]
        img = Image.open(img_file).convert('RGB')
        img = img.resize((256, 256))
        np3ch = np.array(img)

        # If image is grayscale, convert to "color" by replicating channels
        if np3ch.shape != (256, 256, 3):
            np3ch = np3ch.repeat(3,1,1)

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        label = label.resize((256, 256))
        label = np.asarray(label)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        # Permute to make image channel first
        np3ch = np.moveaxis(np3ch, -1, 0)
        return np3ch, label, domain
        '''
        raise NotImplementedError




