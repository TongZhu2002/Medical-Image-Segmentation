import os
import zarr
from lightning import Trainer
import numpy as np
import torch
from data_loader import MRIDataset
from train import BraTSNet
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader


def main():
    print("Testing data_loader.py")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    # Train 1251
    # Train:Val:Test = (8:2:1) = (1000:250:125)
    ratio = (8, 2, 1)
    total = len(zarr.open("data/samples.zarr", mode="r"))
    first, last = (
        int(total * ratio[0] / sum(ratio)),
        int(total * ratio[2] / sum(ratio)),
    )

    total = np.arange(total)
    np.random.shuffle(total)
    train = total[:first]
    val = total[first:-last]
    test = total[-last:]

    train_dataset = MRIDataset(
        samples_file="data/samples.zarr", labels_file="data/labels.zarr", indeces=train
    )
    val_dataset = MRIDataset(
        samples_file="data/samples.zarr", labels_file="data/labels.zarr", indeces=val
    )
    # test_dataset = MRIDataset(
    #     samples_file="data/samples.zarr", labels_file="data/labels.zarr", indeces=test
    # )

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
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=1,
    #     # pin_memory=True,
    # )

    model = BraTSNet()
    logger = TensorBoardLogger(
        os.path.curdir,
        name="lightning_logs",
        log_graph=False,
    )

    trainer = Trainer(
        accelerator="gpu",
        strategy="fsdp",
        devices="auto",
        precision="bf16-mixed",
        logger=logger,
        callbacks=None,
        max_epochs=100,
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
