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

## Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

JPEG
```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --train /imagenet/train --val /imagenet/val
```

LMDB
* NOTE: Since LMDB can't be pickled, you need to hack the `folder2lmdb.ImageFolderLMDB` to delay the loading of the environment, such as below:

```python
class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        # https://github.com/chainer/chainermn/issues/129
	# Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = None
	
	# Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time
	if 'train' in self.db_path:
            self.length = 1281167
        elif 'val' in self.db_path:
            self.length = 50000
        else:
            raise NotImplementedError
	...
	
    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
	...
```

Now we can launch LMDB version with `torch.multiprocessing` using above workaround:
```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --train /imagenet/train-lmdb --val /imagenet/val-lmdb --lmdb
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

