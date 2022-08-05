import os

import torch.utils.data as data
from PIL import Image


class TrashSegmentation(data.Dataset):
    def __init__(self, root_dir, transforms=None, txt_name: str = "train.txt"):
        super(TrashSegmentation, self).__init__()
        image_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'segmentationMask')
        assert os.path.exists(mask_dir), "path '{}' does not exist.".format(mask_dir)
        assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)

        file_names = [f for f in os.listdir(mask_dir) if not f.startswith('.')]
        file_names = [int(name[:-4]) for name in file_names]
        file_names = sorted(file_names)
        
        self.images = [os.path.join(image_dir, str(x) + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, str(x) + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
