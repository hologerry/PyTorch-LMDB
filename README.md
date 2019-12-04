# PyTorch-LMDB

Scripts to work with LMDB + PyTorch for Imagenet training

> **NOTE**: This has only been tested in the [NGC PyTorch 19.11-py3 container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
>
> Other environments have not been tested

Much of this code and LMDB documentation was adopted from https://github.com/Lyken17/Efficient-PyTorch, so credits to @Lyken17.

## Quickstart

1. Start interactive PyTorch container

```bash
nvidia-docker run -it -v ${PWD}:/mnt -v /imagenet:/imagenet --workdir=/mnt nvcr.io/nvidia/pytorch:19.11-py3
```

2. Convert data to LMDB format

```bash
mkdir -p train-lmdb/
python folder2lmdb.py --dataset /imagenet/train -o train-lmdb/

mkdir -p val-lmdb/
python folder2lmdb.py --dataset /imagenet/val -o val-lmdb/
```

3. Run training on LMDB data

```bash
time python main.py --arch resnet50 --train train-lmdb/ --val val-lmdb/ --lmdb --epochs 2
```

4. (Optional) Compare to JPEG data

```bash
time python main.py --arch resnet50 --train /imagenet/train --val /imagenet/val --epochs 2
```

## LMDB

LMDB is a json-like, but in binary stream key-value storage. In my design, the format of converted LMDB is defined as follow.

key | value 
--- | ---
img-id1 | (jpeg_raw1, label1)
img-id2 | (jpeg_raw2, label2)
img-id3 | (jpeg_raw3, label3)
... | ...
img-idn | (jpeg_rawn, labeln)
`__keys__` | [img-id1, img-id2, ... img-idn]
`__len__` | n

As for details of reading/writing, please refer to [code](folder2lmdb.py).

### LMDB Dataset / DataLoader

`folder2lmdb.py` has an implementation of a PyTorch `ImageFolder` for LMDB data to be passed into the `torch.utils.data.DataLoader`.

In `main.py`, passing the `--lmdb` flag specifies to use `folder2lmdb.ImageFolderLMDB` instead of the default
`torchvision.datasets.ImageFolder` when setting up the data. 

```python
# Data loading code
if args.lmdb:
    import folder2lmdb
    image_folder = folder2lmdb.ImageFolderLMDB
else:
    image_folder = datasets.ImageFolder

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
				 std=[0.229, 0.224, 0.225])

train_dataset = image_folder(
    args.train,
    transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	normalize,
    ]))

val_dataset = image_folder(
    args.val, 
    transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	normalize,
    ]))
```

`ImageFolderLMDB` can be simply used in place of `ImageFolder` like below:

```python
from folder2lmdb import ImageFolderLMDB
from torch.utils.data import DataLoader
dataset = ImageFolderLMDB(path, transform, target_transform)
loader = DataLoader(dataset, batch_size=64)
```

