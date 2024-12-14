from collections import OrderedDict
from typing import Callable, Optional, Sequence, Union

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor, nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics_dict: dict[str, Sequence[Callable[[Tensor, Tensor], float]]],
        clf_keys: set[str],
        reg_keys: set[str],
    ):
        """
        Initialize the LightningModule.

        Args:
            model: The model to train.
            criterion: The loss function.
            metrics: Dictionary of metrics to compute.
            clf_keys: Set of classification keys.
            reg_keys: Set of regression keys.
        """

        super().__init__()
        self.model = model
        self.clf_keys, self.reg_keys = clf_keys, reg_keys
        self.criterion = criterion
        self.metrics_dict = metrics_dict

    def _step(
        self,
        batch: tuple[Tensor, dict[str, Tensor], dict[str, Tensor]],
        mode: str,
    ) -> Tensor:
        """
        Perform a single step.
        Args:
            batch: The batch.
            mode: The mode 'train' | 'val' | 'test' | ...
        Returns:
            The prediction and the loss.
        """
        input, target, _ = batch

        output = self.model(input)

        target = LightningModel._reshape_to_match(target, output, self.reg_keys)

        loss = LightningModel._aggregate_loss(self.compute_loss(output, target))

        metrics_dict = self.compute_metrics(output, target)
        self.log_metrics(loss, metrics_dict, mode)

        return output, loss

    @staticmethod
    def _reshape_to_match(
        tensors_to_reshape: dict[str, Tensor],
        tensors_to_match: dict[str, Tensor],
        keys: Optional[set[str]] = None,
    ) -> dict[str, Tensor]:
        """
        Reshape the tensors to match the shape of the target tensors.

        Args:
            tensors_to_reshape: Dictionary of tensors to reshape.
            tensors_to_match: Dictionary of tensors to match.
            keys: The keys to match.

        Returns:
            Dictionary of reshaped tensors.
        """

        keys = keys or (tensors_to_match.keys() & tensors_to_reshape.keys())

        return {
            key: (
                tensors_to_reshape[key].reshape_as(tensors_to_match[key])
                if key in keys
                and tensors_to_reshape[key].shape != tensors_to_match[key].shape
                else tensors_to_reshape[key]
            )
            for key in tensors_to_reshape.keys()
        }

    @staticmethod
    def _aggregate_loss(
        loss_dict: dict[str, Tensor], weights: dict[str, float] = {}
    ) -> Tensor:
        """
        Aggregate the loss components into a single scalar loss.

        Args:
            loss_dict: Dictionary of loss components.
            weights: Optional dictionary of weights for each loss component.

        Returns:
            Aggregated scalar loss.
        """
        return torch.stack(
            [weights.get(key, 1.0) * loss_dict[key] for key in loss_dict.keys()]
        ).sum()

    def _prepare_target_tensors(
        self,
        target: dict[str, Tensor],
        output: OrderedDict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Prepare the target tensors to match the output tensors.

        Args:
            target: Dictionary of target tensors.
            output: Dictionary of output tensors.

        Returns:
            Dictionary of target tensors.
        """
        prepared_target = {
            key: target[key].clone(memory_format=torch.preserve_format).detach()
            for key in target.keys()
        }

        for branch in self.clf_keys:
            if target[branch].ndim < output[branch].ndim:
                prepared_target[branch] = (
                    F.one_hot(
                        target[branch].long(),
                        num_classes=output[branch].shape[1],
                    )
                    .float()
                    .permute(0, 3, 1, 2)
                )

        return LightningModel._reshape_to_match(prepared_target, output)

    def compute_loss(
        self, output: OrderedDict[str, Tensor], target: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Compute the loss function.

        Args:
            output: The model output.
            target: The target tensors.

        Returns:
            The loss for each output key.
        """

        prepared_target = self._prepare_target_tensors(target, output)

        return self.criterion(output, prepared_target)

    def compute_metrics(
        self,
        prediction: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> dict[str, float]:
        """
        Compute the metrics.

        Args:
            prediction: The model prediction.
            target: The target tensors.

        Returns:
            The metrics.
        """

        return {
            "_".join((metr.__name__, key)): metr(prediction[key], target[key]).item()
            for key, metrs in self.metrics_dict.items()
            for metr in metrs
        }

    def log_metrics(self, loss: float, metrics_dict: dict[str, float], mode: str):
        """
        Log the metrics.

        Args:
            loss: The loss value.
            metrics: The metrics.
            mode: The mode (train or val).
        """

        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            {f"{mode}_{key}": val for key, val in metrics_dict.items()},
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(
        self,
    ) -> dict[str, Union[Optimizer, LRScheduler]]:
        """
        Configure the optimizer and the learning rate scheduler.

        Returns:
            The optimizer and the learning rate scheduler.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-3,
                total_steps=self.trainer.estimated_stepping_batches,
            ),
        }

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Perform a training step.
        Args:
            batch: The batch.
            batch_idx: The batch index.

        Returns:
            The loss value.
        """

        _, loss = self._step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx) -> OrderedDict[str, Tensor]:
        """
        Perform a validation step.

        Args:
            batch: The batch.
            batch_idx: The batch index.
        """

        output, _ = self._step(batch, mode="val")
        return output

    def test_step(self, batch, batch_idx) -> OrderedDict[str, Tensor]:
        """
        Perform a test step.

        Args:
            batch: The batch.
            batch_idx: The batch index.
        """

        output, _ = self._step(batch, mode="test")
        return output
