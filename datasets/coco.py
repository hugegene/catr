from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
import cv2

from transformers import BertTokenizer

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)

class Fogging:
    def __init__(self,):
        self.seq = iaa.Sequential([iaa.CloudLayer([190,255], [-2.5,1.5], [0,10], [0, 0.3], [0,0.5], 1, -1.5, 1, 1)], random_order=True)

    def __call__(self, x):
        in_image = x.convert('RGB')
        img_np = np.asarray(in_image)
        open_cv_image = img_np[:, :, ::-1].copy()
        augimage = self.seq(images=[open_cv_image])[0]
        img = cv2.cvtColor(augimage, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        return im_pil

train_transform = tv.transforms.Compose([
    #Fogging(),
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        # val = str(image_id).zfill(12)
        val = str(image_id)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        # image = image = cv2.imread(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


# def build_dataset(config, mode='training'):
#     if mode == 'training':
#         train_dir = os.path.join(config.dir, 'train2017')
#         train_file = os.path.join(
#             config.dir, 'annotations', 'captions_train2017.json')
#         data = CocoCaption(train_dir, read_json(
#             train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
#         return data

#     elif mode == 'validation':
#         val_dir = os.path.join(config.dir, 'val2017')
#         val_file = os.path.join(
#             config.dir, 'annotations', 'captions_val2017.json')
#         data = CocoCaption(val_dir, read_json(
#             val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
#         return data

#     else:
#         raise NotImplementedError(f"{mode} not supported")
    

def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir)
        train_file = os.path.join(
            config.dir, 'annotations', 'captions_train.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir)
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_val.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")

