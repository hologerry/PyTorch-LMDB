import os
import lmdb
import torch  # noqa
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
import pyarrow as pa
import six
from PIL import Image


class ImageAttributeLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        label = torch.FloatTensor(target)
        return {
            "img_a": img, "attr_a": label
        }

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


if __name__ == "__main__":
    db_path = '/D_data/Face_Editing/face_editing/data/celebahq_lmdb/train'
    transforms = [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    trans = T.Compose(transforms)
    dest = ImageAttributeLMDB(db_path, trans)
    dloader = DataLoader(dest, batch_size=2, shuffle=True)
    for i, item in enumerate(dloader):
        if i > 0:
            break
        img = item["img_a"]
        lbl = item["attr_a"]
        if i == 0:
            print("img size", img.size())
            print("label size", lbl.size())
            from torchvision.utils import save_image
            save_image(img, "saved_img.png", normalize=True)
            print(lbl)
