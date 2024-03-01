from functools import reduce
import gc
import nibabel as nib
import os
from glob import glob
import torch
import zarr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed, cpu_count
from memory_profiler import profile


def fit_to_minmax(img, scaler):
    return scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)


def calculate_weights(seg_img):
    """
    Labels:
    - 1 for NCR
    - 2 for ED
    - 3 for ET
    - 0 for everything else.
    """
    # times each shape
    voxel_count = reduce(lambda x, y: x * y, seg_img.shape)
    ET = len(np.where(seg_img == 3)[0]) / voxel_count
    ED = len(np.where(seg_img == 2)[0]) / voxel_count
    NCR = len(np.where(seg_img == 1)[0]) / voxel_count
    BG = len(np.where(seg_img == 0)[0]) / voxel_count
    # calculate weights
    return (ET, ED, NCR, BG)


def get_sample(args):
    i, file, dataset_folder = args
    file = file.split("/")[-1]
    print("Processing file: ", file, "(", i + 1, ")")
    flair = os.path.join(dataset_folder, file, file + "-t2w.nii.gz")
    seg = os.path.join(dataset_folder, file, file + "-seg.nii.gz")
    t1 = os.path.join(dataset_folder, file, file + "-t1n.nii.gz")
    t1ce = os.path.join(dataset_folder, file, file + "-t1c.nii.gz")
    t2 = os.path.join(dataset_folder, file, file + "-t2f.nii.gz")

    # samples
    # 240 [55:55+128]
    # 240 [63:63+128]
    # 155 [12:12+128]
    x = slice(55, 55 + 128, 1)
    y = slice(63, 63 + 128, 1)
    z = slice(12, 12 + 128, 1)
    flair_img = nib.load(flair).get_fdata()[x, y, z]
    t1_img = nib.load(t1).get_fdata()[x, y, z]
    t1ce_img = nib.load(t1ce).get_fdata()[x, y, z]
    t2_img = nib.load(t2).get_fdata()[x, y, z]

    scaler = MinMaxScaler()
    flair_img = fit_to_minmax(flair_img, scaler)
    t1_img = fit_to_minmax(t1_img, scaler)
    t1ce_img = fit_to_minmax(t1ce_img, scaler)
    t2_img = fit_to_minmax(t2_img, scaler)
    X = np.stack((flair_img, t1_img, t1ce_img, t2_img), axis=-1)
    # labels
    seg_img = nib.load(seg).get_fdata().astype(np.uint8)[x, y, z]
    seg_img_weights = calculate_weights(seg_img)
    Y = torch.nn.functional.one_hot(
        torch.tensor(seg_img, dtype=torch.long), num_classes=4
    )
    # samples[i] = X
    # labels[i] = Y
    # ET, ED, NCR, BG = seg_img_weights
    # indeces_map[i] = [file, ET, ED, NCR, BG]
    # return i, file, seg_img_weights

    del flair_img, t1_img, t1ce_img, t2_img, seg_img, scaler, flair, t1, t1ce, t2, seg
    return i, file, X, Y, seg_img_weights


@profile
def data_preprocessing(
    dataset_folder,
    samples_file="data/samples.zarr",
    labels_file="data/labels.zarr",
    indeces_map_file="data/indeces_map.csv",
    partial_dataset=None,
):
    """_summary_

    Args:
        dataset_folder (_type_): _description_
        samples_file (str, optional): _description_. Defaults to "data/samples.zarr".
        labels_file (str, optional): _description_. Defaults to "data/labels.zarr".
        indeces_map_file (str, optional): _description_. Defaults to "data/indeces_map.csv".
        partial_dataset (optional): like `glob(dataset_folder + "/*")`. Defaults to None.
    """
    # samples
    # (index, x, y, z, channels)
    # samples = zarr.empty((1, 240, 240, 155, 4), chunks=True)
    total = len(glob(dataset_folder + "/*"))
    samples = zarr.open(
        samples_file,
        mode="w",
        # shape=(total, 240, 240, 155, 4),
        # chunks=(1, 240, 240, 155, 4),
        shape=(total, 128, 128, 128, 4),
        chunks=(1, 128, 128, 128, 4),
        dtype="float64",
    )
    # labels
    # (index, x, y, z, classes)
    # labels = zarr.empty((1, 240, 240, 155, 4), chunks=True)
    labels = zarr.open(
        labels_file,
        mode="w",
        # shape=(total, 240, 240, 155, 4),
        # chunks=(1, 240, 240, 155, 4),
        shape=(total, 128, 128, 128, 4),
        chunks=(1, 128, 128, 128, 4),
        dtype="float64",
    )

    # file = "BraTS-GLI-00000-000"

    indeces_map = [""] * total

    # calc run time
    import time

    start = time.time()
    # def get_sample(i, file) function

    print(cpu_count())
    if not partial_dataset:
        partial_dataset = glob(dataset_folder + "/*")

    results = Parallel(
        # n_jobs=18, verbose=10, return_as="generator", backend="threading"
        n_jobs=18,
        return_as="generator",
        backend="threading",
    )(
        delayed(get_sample)((i, file, dataset_folder))
        for i, file in enumerate(partial_dataset)
    )

    # return_as="list"
    # results = [1,2,3...]
    # return_as="generator"
    # results = iterator
    # 每次调用迭代器的next()方法时，就会返回一个结果
    # next(iterator) 1 2 3 ...

    # (30mb+10mb)*1250 = 50gb
    # 24GB RAM

    def store_results(result):
        i, file, X, Y, seg_img_weights = result
        samples[i] = X
        labels[i] = Y
        ET, ED, NCR, BG = seg_img_weights
        indeces_map[i] = [file, ET, ED, NCR, BG]
        del X, Y, ET, ED, NCR, BG
        gc.collect()

    Parallel(n_jobs=6, return_as="list", backend="threading")(
        delayed(store_results)(result) for result in results
    )

    # for i, file, seg_img_weights in results:
    # for i, file, X, Y, seg_img_weights in results:
    #     samples[i] = X
    #     labels[i] = Y
    #     ET, ED, NCR, BG = seg_img_weights
    #     indeces_map[i] = [file, ET, ED, NCR, BG]
    #     del X, Y, ET, ED, NCR, BG
    #     gc.collect()

    end = time.time()
    df = pd.DataFrame(
        indeces_map,
        columns=["filename", "ET", "ED", "NCR", "BG"],
    )
    df.to_csv(indeces_map_file, index=True)

    print("Time elapsed: ", round(end - start, 2), "s")


if __name__ == "__main__":
    # Train 1251
    # Train:Val:Test = (8:2:1) = (1000:250:125)
    ratio = (8, 2, 1)
    total = 1251
    first, last = (
        int(total * ratio[0] / sum(ratio)),
        int(total * ratio[2] / sum(ratio)),
    )
    print(first, last)

    folder = "data/BraTS2023-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"  # train
    # folder = "data/BraTS2023-dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"  # val -> val/test
    data_preprocessing(
        dataset_folder=folder,
        samples_file="data/val_samples.zarr",
        labels_file="data/val_labels.zarr",
        indeces_map_file="data/val_indeces_map.csv",
        partial_dataset=glob(folder + "/*"),
    )
