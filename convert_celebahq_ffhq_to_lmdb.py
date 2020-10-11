import os
from os.path import join as ospj
import random

import lmdb

from folder2lmdb import raw_reader
from folder2lmdb import dumps_pyarrow


def write_list_to_lmdb(dataset_list, lmdb_path, write_frequency=1000):
    os.makedirs(lmdb_path, exist_ok=True)
    isdir = os.path.isdir(lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx, data in enumerate(dataset_list):
        img_path = data[0]
        image = raw_reader(img_path)
        label = data[1]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(dataset_list)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(dataset_list))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def celebahq2lmdb(celebahq_path, outpath, selected_attrs):
    print("Processing CelebAHQ dataset ...")
    celeba_attr_file = ospj(celebahq_path, 'CelebAMask-HQ-attribute-anno-skin.txt')
    img_path = ospj(celebahq_path, 'CelebA-HQ-img-256x256')
    with open(celeba_attr_file, 'r') as f:
        img_name_attrs_lines = f.readlines()

    attr2idx = {}
    idx2attr = {}

    all_attr_names = img_name_attrs_lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = img_name_attrs_lines[2:]
    random.seed(1234)
    random.shuffle(lines)

    celeba_test_dataset = []
    celeba_train_dataset = []

    for i, line in enumerate(lines):
        split = line.strip().split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')

        filepath = ospj(img_path, filename)
        if i < 2000:
            celeba_test_dataset.append([filepath, label])
        else:  # 28000
            celeba_train_dataset.append([filepath, label])

    assert len(celeba_train_dataset) == 28000
    assert len(celeba_test_dataset) == 2000

    celeba_eval_part1 = celeba_test_dataset[:len(celeba_test_dataset)//2]
    celeba_eval_part2 = celeba_test_dataset[len(celeba_test_dataset)//2:]
    assert len(celeba_eval_part1) == 1000
    assert len(celeba_eval_part2) == 1000

    train_lmdb_dir = ospj(outpath, 'train')
    write_list_to_lmdb(celeba_train_dataset, train_lmdb_dir)

    test_lmdb_dir = ospj(outpath, 'test')
    write_list_to_lmdb(celeba_test_dataset, test_lmdb_dir)
    eval_part1_lmdb_dir = ospj(outpath, 'eval_part1')
    write_list_to_lmdb(celeba_eval_part1, eval_part1_lmdb_dir)
    eval_part2_lmdb_dir = ospj(outpath, 'eval_part2')
    write_list_to_lmdb(celeba_eval_part2, eval_part2_lmdb_dir)
    print("Finished processing CelebAHQ dataset ...")
    return celeba_train_dataset, celeba_test_dataset, celeba_eval_part1, celeba_eval_part2


def ffhq2lmdb(ffhq_path, outpath, selected_attrs):
    print("Processing FFHQ dataset ...")
    attr_file = ospj(ffhq_path, 'ffhq_attributes_list.txt')
    img_path = ospj(ffhq_path, 'images256x256')
    with open(attr_file, 'r') as f:
        img_name_attrs_lines = f.readlines()

    attr2idx = {}
    idx2attr = {}

    all_attr_names = img_name_attrs_lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    ffhq_test_dataset = []
    ffhq_train_dataset = []

    lines = img_name_attrs_lines[2:]
    for i, line in enumerate(lines):
        split = line.strip().split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')
        filepath = ospj(img_path, filename)
        if i >= 66000:
            ffhq_test_dataset.append([filepath, label])
        else:  # 4000
            ffhq_train_dataset.append([filepath, label])

    assert(len(ffhq_train_dataset)) == 66000
    assert(len(ffhq_test_dataset)) == 4000

    ffhq_eval_part1 = ffhq_test_dataset[:len(ffhq_test_dataset)//2]
    ffhq_eval_part2 = ffhq_test_dataset[len(ffhq_test_dataset)//2:]
    assert len(ffhq_eval_part1) == 2000
    assert len(ffhq_eval_part2) == 2000

    train_lmdb_dir = ospj(outpath, 'train')
    write_list_to_lmdb(ffhq_train_dataset, train_lmdb_dir)

    test_lmdb_dir = ospj(outpath, 'test')
    write_list_to_lmdb(ffhq_test_dataset, test_lmdb_dir)
    eval_part1_lmdb_dir = ospj(outpath, 'eval_part1')
    write_list_to_lmdb(ffhq_eval_part1, eval_part1_lmdb_dir)
    eval_part2_lmdb_dir = ospj(outpath, 'eval_part2')
    write_list_to_lmdb(ffhq_eval_part2, eval_part2_lmdb_dir)
    print("Finished processing FFHQ dataset")
    return ffhq_train_dataset, ffhq_test_dataset, ffhq_eval_part1, ffhq_eval_part2


def celebahq_ffhq_to_lmdb(celebahq_ffhq_fake_outpath, train, test, eval_part1, eval_part2):
    print("Processing CelebaHQ_FFHQ dataset ...")
    train_dir = ospj(celebahq_ffhq_fake_outpath, 'train')
    test_dir = ospj(celebahq_ffhq_fake_outpath, 'test')
    eval_part1_dir = ospj(celebahq_ffhq_fake_outpath, 'eval_part1')
    eval_part2_dir = ospj(celebahq_ffhq_fake_outpath, 'eval_part2')
    write_list_to_lmdb(train, train_dir)
    write_list_to_lmdb(test, test_dir)
    write_list_to_lmdb(eval_part1, eval_part1_dir)
    write_list_to_lmdb(eval_part2, eval_part2_dir)
    print("Finished processing CelebaHQ_FFHQ dataset ...")


if __name__ == "__main__":
    celebahq_path = '/D_data/Face_Editing/face_editing/data/celebahq'
    celebahq_outpath = '/D_data/Face_Editing/face_editing/data/celebahq_lmdb'
    ffhq_path = '/D_data/Face_Editing/face_editing/data/ffhq'
    ffhq_outpath = '/D_data/Face_Editing/face_editing/data/ffhq_lmdb'
    celebahq_ffhq_fake_outpath = '/D_data/Face_Editing/face_editing/data/celebahq_ffhq_fake_lmdb'
    selected_attrs = ['Arched_Eyebrows', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses',
                      'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                      'No_Beard', 'Smiling', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
    celeba_train, celeba_test, celeba_evalp1, celeba_evalp2 = celebahq2lmdb(celebahq_path, celebahq_outpath, selected_attrs)
    ffhq_train, ffhq_test, ffhq_evalp1, ffhq_evalp2 = ffhq2lmdb(ffhq_path, ffhq_outpath, selected_attrs)
    train = celeba_train + ffhq_train
    assert len(train) == 28000 + 66000
    test = celeba_test + ffhq_test
    assert len(test) == 2000 + 4000
    evalp1 = celeba_evalp1 + ffhq_evalp1
    assert len(evalp1) == 1000 + 2000
    evalp2 = celeba_evalp2 + ffhq_evalp2
    assert len(evalp2) == 1000 + 2000
    celebahq_ffhq_to_lmdb(celebahq_ffhq_fake_outpath, train, test, evalp1, evalp2)
