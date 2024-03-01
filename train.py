import torch
import torch.nn as nn
import torch.optim as optim
from unet3d import Unet3D
import lightning.pytorch as pl
from torchmetrics.classification import Dice
from monai.metrics import compute_percent_hausdorff_distance
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


# define the LightningModule
# Trainer
class BraTSNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Unet3D()
        self.example_input_array = torch.rand(1, 4, 128, 128, 128)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # metrics
        self.dice_score_c1 = Dice(average="micro")
        self.dice_score_c2 = Dice(average="micro")
        self.dice_score_c3 = Dice(average="micro")
        self.dice_score_c4 = Dice(average="micro")
        self.hasudorff_dis = compute_percent_hausdorff_distance

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=1e-4)
        return {
            "optimizer": optimizer,
        }

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val/loss", mode="min", patience=10, min_delta=0.001
        )
        checkpoint = ModelCheckpoint(
            monitor="val/dice", save_last=True, save_top_k=1, mode="min"
        )
        return [early_stop, checkpoint]

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        s_x = slice(55, 55 + 128, 1)
        s_y = slice(63, 63 + 128, 1)
        s_z = slice(12, 12 + 128, 1)
        x, y = x[..., s_x, s_y, s_z], y.permute(0, 4, 1, 2, 3)[..., s_x, s_y, s_z]
        y_hat = self(x)
        c1 = 1 - self.dice_score_c1(
            y_hat[:, 0, ...].flatten(1), y[:, 0, ...].to(torch.long).flatten(1)
        )
        c2 = 1 - self.dice_score_c2(
            y_hat[:, 1, ...].flatten(1), y[:, 1, ...].to(torch.long).flatten(1)
        )
        c3 = 1 - self.dice_score_c3(
            y_hat[:, 2, ...].flatten(1), y[:, 2, ...].to(torch.long).flatten(1)
        )
        c4 = 1 - self.dice_score_c4(
            y_hat[:, 3, ...].flatten(1), y[:, 3, ...].to(torch.long).flatten(1)
        )
        loss = self.loss_fn(y_hat, y) + 0.1 * ((c1 + c2 + c3 + c4) / 4)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/dice",
            (c1 + c2 + c3 + c4) / 4,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        s_x = slice(55, 55 + 128, 1)
        s_y = slice(63, 63 + 128, 1)
        s_z = slice(12, 12 + 128, 1)
        x, y = x[..., s_x, s_y, s_z], y.permute(0, 4, 1, 2, 3)[..., s_x, s_y, s_z]
        y_hat = self(x)
        c1 = 1 - self.dice_score_c1(
            y_hat[:, 0, ...].flatten(1), y[:, 0, ...].to(torch.long).flatten(1)
        )
        c2 = 1 - self.dice_score_c2(
            y_hat[:, 1, ...].flatten(1), y[:, 1, ...].to(torch.long).flatten(1)
        )
        c3 = 1 - self.dice_score_c3(
            y_hat[:, 2, ...].flatten(1), y[:, 2, ...].to(torch.long).flatten(1)
        )
        c4 = 1 - self.dice_score_c4(
            y_hat[:, 3, ...].flatten(1), y[:, 3, ...].to(torch.long).flatten(1)
        )
        loss = self.loss_fn(y_hat, y) + 0.1 * ((c1 + c2 + c3 + c4) / 4)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/dice",
            (c1 + c2 + c3 + c4) / 4,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # y_hat = torch.nn.functional.one_hot(
        #     torch.tensor(y_hat.detach().cpu().argmax(1), dtype=torch.long), num_classes=4
        # ).permute(0,4,1,2,3).numpy()
        # y = y.detach().cpu().to(torch.bool).numpy()
        # s = slice(0,32)
        # self.log(
        #     "val/hasudorff_dis",
        #     self.hasudorff_dis(y_hat[0:2,0:2,s,s,s], y[0:2,0:2,s,s,s], percentile=95),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
