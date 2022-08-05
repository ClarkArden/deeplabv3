import os
import random
import shutil
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

json_path = "./final/1425_annotations.json"
img_path = "./final/images"
mask_path = "./final/segmentationMask"
palette_path = "./palette.json"
with open(palette_path, "rb") as f:
    pallette_dict = json.load(f)
    pallette = []
    for v in pallette_dict.values():
        pallette += v

# random.seed(0)
# pallette = [0, 0, 0] + [random.randint(0, 255) for _ in range(255*3)]

# load coco data
coco = COCO(annotation_file=json_path)

# print coco info
coco.info()

filename = [f for f in os.listdir(img_path) if not f.startswith('.')]
filename = [int(name[:-4]) for name in filename]
ids1 =sorted(filename)


# get all image index info
imgtoann = coco.imgToAnns
ids2 = list(imgtoann.keys())

ids = list(set(ids1) & set(ids2))
diff1 = list(set(ids1) ^ set(ids))      #有图像数据但没有标注数据的图像
diff2 = list(set(ids2) ^ set(ids))      #有标注数据但没有图像数据



print("number of images: {}".format(len(ids)))

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

print('H')
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def mask_generator1(width, height, targets):
    masks = []
    cats = []
    for target in targets:
        cats.append(target["category_id"])  # get object class id
        polygons = target["segmentation"]   # get object polygons
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = mask.any(axis=2)
        masks.append(mask)

    cats = np.array(cats, dtype=np.int32)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    # merge all instance masks into a single segmentation map
    # with its corresponding categories
    target = (masks * cats[:, None, None]).max(axis=0)
    # discard overlapping instances
    target[masks.sum(0) > 1] = 255
    target = Image.fromarray(target.astype(np.uint8))

    target.putpalette(pallette)
    return target

# 生成mask图
def mask_generator(width, height, anns_list):
    mask_pic = np.zeros((height, width))
    # 生成mask - 此处生成的是4通道的mask图,如果使用要改成三通道,可以将下面的注释解除,或者在使用图片时搜相关的程序改为三通道
    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask_pic += mask_single
    # 转化为255
    # for row in range(height):
    #     for col in range(width):
    #         if (mask_pic[row][col] > 0):
    #             mask_pic[row][col] = 255
    # mask_pic = mask_pic.astype(int)
    # return mask_pic

    # 转为三通道
    imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
    imgs[:, :, 0] = mask_pic[:, :]
    imgs[:, :, 1] = mask_pic[:, :]
    imgs[:, :, 2] = mask_pic[:, :]
    imgs = imgs.astype(np.uint8)
    target = Image.fromarray(imgs).convert('P')
    target.putpalette(pallette)
    return target

mkr(mask_path)

for img_id in ids:
    img = coco.loadImgs(img_id)[0]
    img_w = img['width']
    img_h = img['height']
    print("img_id={} img_w={} img_h={}".format(img_id, img_w, img_h))
    
    # 获取对应图像id的所有annotations idx信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)
    mask_img = mask_generator1(img_w, img_h, targets)
    save_img = mask_img.convert('RGB')
    
    img_name = str(img_id) + '.png'
    save_path = os.path.join(mask_path, img_name)
    # plt.imsave(save_path, mask_img)
    mask_img.save(save_path)
    
    
    