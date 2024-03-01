import torch
import zarr
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path


class MRIDataset(Dataset):
    def __init__(self, samples_file, labels_file, indeces):
        self.samples_file = Path(samples_file)
        self.labels_file = Path(labels_file)
        self.indeces = indeces

    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, idx):
        idx = self.indeces[idx]
        samples = zarr.open(self.samples_file, mode="r")
        labels = zarr.open(self.labels_file, mode="r")
        X = torch.tensor(samples[idx]).permute(3, 0, 1, 2)
        Y = torch.tensor(labels[idx])
        return X, Y


if __name__ == "__main__":
    print("Testing data_loader.py")

    import numpy as np

    # Train 1251
    # Train:Val:Test = (8:2:1) = (1000:250:125)
    ratio = (8, 2, 1)
    total = len(zarr.open("data/samples.zarr", mode="r"))
    first, last = (
        int(total * ratio[0] / sum(ratio)),
        int(total * ratio[2] / sum(ratio)),
    )
    # print(first, last)

    total = np.arange(total)
    np.random.shuffle(total)
    # print(total)

    train = total[:first]
    val = total[first:-last]
    test = total[-last:]
    # print(len(train), len(val), len(test))
    # print(train)
    # print(val)
    # print(test)

    train_dataset = MRIDataset(
        samples_file="data/samples.zarr", labels_file="data/labels.zarr", indeces=train
    )
    val_dataset = MRIDataset(
        samples_file="data/samples.zarr", labels_file="data/labels.zarr", indeces=val
    )
    test_dataset = MRIDataset(
        samples_file="data/samples.zarr", labels_file="data/labels.zarr", indeces=test
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        # pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        # pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        # pin_memory=True,
    )

    # for item in range(len(dataset)):
    #     print(f"Length of sample tuple: {len(dataset[item])}")
    #     print(f"Shape of sample: {dataset[item][0].shape}")
    #     print(f"Shape of sample: {dataset[item][1].shape}")
    limit = 2

    for i, batch in enumerate(train_dataloader):
        if i > limit:
            break
        print(f"train Batch shape: {batch[0].shape} {i}")

    for i, batch in enumerate(val_dataloader):
        if i > limit:
            break
        print(f"val Batch shape: {batch[0].shape} {i}")

    for i, batch in enumerate(test_dataloader):
        if i > limit:
            break
        print(f"test Batch shape: {batch[0].shape} {i}")
