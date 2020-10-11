import os
import PIL
from PIL import Image
from os.path import join as ospj

target_img_size = 256

celebahq_orgin_dir = '/D_data/Face_Editing/face_editing/data/celebahq/CelebA-HQ-img'
celebahq_target_dir = '/D_data/Face_Editing/face_editing/data/celebahq/CelebA-HQ-img-256x256'
os.makedirs(celebahq_target_dir, exist_ok=True)


for i in range(30000):
    filepath = ospj(celebahq_orgin_dir, f'{i}.jpg')
    img = Image.open(filepath)
    img = img.convert('RGB')
    img = img.resize((target_img_size, target_img_size), resample=PIL.Image.BILINEAR)
    tgtfilepath = ospj(celebahq_target_dir, f'{i}.jpg')
    img.save(tgtfilepath, quality=100, subsampling=0)


ffhq_orgin_dir = '/D_data/Face_Editing/face_editing/data/ffhq/images1024x1024'
ffhq_target_dir = '/D_data/Face_Editing/face_editing/data/ffhq/images256x256'
os.makedirs(ffhq_target_dir, exist_ok=True)

for i in range(70000):
    img_sub_dir = f'{(i // 1000):02d}000'
    filepath = ospj(ffhq_orgin_dir, img_sub_dir, f'{i:05d}.png')
    img = Image.open(filepath)
    img = img.convert('RGB')
    img = img.resize((target_img_size, target_img_size), resample=PIL.Image.BILINEAR)
    tgtfilepath = ospj(ffhq_target_dir, f'{i:05d}.png')
    img.save(tgtfilepath)
