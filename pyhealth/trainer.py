import logging
import os
from datetime import datetime
from typing import Dict, List, Type, Callable
from typing import Optional
import pickle
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange

from pyhealth.metrics import (
    binary_metrics_fn,
    multiclass_metrics_fn,
    multilabel_metrics_fn,
)
from pyhealth.utils import create_directory
from .models.attack.sift import *
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0,
                 path='checkpoint.pt', trace_func=print,
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, monitor_criterion_ES= "max"):
        self.monitor_criterion_ES = monitor_criterion_ES
        if self.monitor_criterion_ES == "max":
            score = val_loss
        elif self.monitor_criterion_ES == "min":
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")


def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return


def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")

class Trainer:
    """Trainer for PyTorch models.

    Args:
        model: PyTorch model.
        checkpoint_path: Path to the checkpoint. Default is None, which means
            the model will be randomly initialized.
        metrics: List of metric names to be calculated. Default is None, which
            means the default metrics in each metrics_fn will be used.
        device: Device to be used for training. Default is None, which means
            the device will be GPU if available, otherwise CPU.
        enable_logging: Whether to enable logging. Default is True.
        output_path: Path to save the output. Default is "./output".
        exp_name: Name of the experiment. Default is current datetime.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        enable_logging: bool = True,
        track_logging: bool = False,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
        model_rf: Optional[nn.Module] = None,
        dataset_name: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name_for_metric = dataset_name
        self.model = model
        self.model_rf = model_rf
        self.metrics = metrics
        self.device = device
        self.early_stopping = EarlyStopping(patience=3, verbose=True)
        self.track_logging = track_logging
        # set logger
        if enable_logging:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_path = os.path.join(output_path, exp_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        # set device
        self.model.to(self.device)

        # logging
        if self.track_logging:
            logger.info(self.model)
            logger.info(f"Metrics: {self.metrics}")
            logger.info(f"Device: {self.device}")

        # load checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)

        logger.info("")
        return

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        epochs: int = 5,
        val_start_epoch: int = 5,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict[str, object]] = None,
        weight_decay: float = 0.0,
        max_grad_norm: float = None,
        monitor: Optional[str] = None,
        monitor_criterion: str = "max",
        load_best_model_at_last: bool = True,
        demo_mode: bool = False,
    ):
        """Trains the model.

        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        if self.track_logging:
            # logging
            logger.info("Training:")
            logger.info(f"Batch size: {train_dataloader.batch_size}")
            logger.info(f"Optimizer: {optimizer_class}")
            logger.info(f"Optimizer params: {optimizer_params}")
            logger.info(f"Weight decay: {weight_decay}")
            logger.info(f"Max grad norm: {max_grad_norm}")
            logger.info(f"Val dataloader: {val_dataloader}")
            logger.info(f"Monitor: {monitor}")
            logger.info(f"Monitor criterion: {monitor_criterion}")
            logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        steps_per_epoch = len(train_dataloader)
        global_step = 0

        # epoch training loop
        show_loss = 'loss nan'
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            #logger.info("")
            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch} {show_loss}",
                smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                output = self.model(**data)
                loss = output["loss"]
                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1
            show_loss = f'loss {sum(training_loss) / len(training_loss):.4f}'

            if demo_mode:
                self.save_ckpt(f"output/normal_epoch{epoch}.ckpt")

            if self.track_logging:
                # log and save
                logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
                logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
                if self.exp_path is not None:
                    self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))
            # validation
            if val_dataloader is not None and epoch > val_start_epoch-1:
                scores = self.evaluate(val_dataloader)
                if self.track_logging:
                    logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                    for key in scores.keys():
                        logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        if self.track_logging:
                            logger.info(
                                f"New best {monitor} score ({score:.4f}) "
                                f"at epoch-{epoch}, step-{global_step}"
                            )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                self.early_stopping(score, monitor_criterion_ES=monitor_criterion)
            if self.early_stopping.early_stop:
                if self.track_logging:
                    print ('Early Stop')
                break

            # load best model
        if load_best_model_at_last and self.exp_path is not None:
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))

        return

    def train_sift(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            val_start_epoch: int = 5,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Optional[Dict[str, object]] = None,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
    ):
        """Trains the model.
        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        steps_per_epoch = len(train_dataloader)
        global_step = 0

        # adv
        adv_modules = hook_sift_layer(self.model, hidden_size=128)
        adv = AdversarialLearner(self.model, adv_modules)

        def logits_fn(model, *wargs, **kwargs):
            output = model(*wargs, **kwargs)
            return output["y_prob"]

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                    steps_per_epoch,
                    desc=f"Epoch {epoch} / {epochs}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                output = self.model(**data)
                loss = output["loss"]
                loss += adv.loss(output["y_prob"], logits_fn, 'bce', **data)

                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1
            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None and epoch > 4:
                scores = self.evaluate(val_dataloader)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                self.early_stopping(score, monitor_criterion_ES=monitor_criterion)

            if self.early_stopping.early_stop:
                print ('Early Stop')
                break

        # load best model
        if load_best_model_at_last and self.exp_path is not None:
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))

        return

    def train_rl_a(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            val_start_epoch: int = 5,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Optional[Dict[str, object]] = None,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
            demo_mode: bool = False,
            reinforcement_confidence: float = 1,
            reinforcement_alpha: float = 0.8,
            reinforcement_beta: float = 0.35,
            loss_decay: float = 0.5,
    ):
        """Trains the model.
        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.

            reinforcement_confidence: the confidence of reward of days
            reinforcement_alpha: the rate of loss and RL_loss
            loss_decay: the loss rate in normal training and RL
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}
        if self.track_logging:
            # logging
            logger.info("Training:")
            logger.info(f"Batch size: {train_dataloader.batch_size}")
            logger.info(f"Optimizer: {optimizer_class}")
            logger.info(f"Optimizer params: {optimizer_params}")
            logger.info(f"Weight decay: {weight_decay}")
            logger.info(f"Max grad norm: {max_grad_norm}")
            logger.info(f"Val dataloader: {val_dataloader}")
            logger.info(f"Monitor: {monitor}")
            logger.info(f"Monitor criterion: {monitor_criterion}")
            logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        steps_per_epoch = len(train_dataloader)
        global_step = 0

        # epoch training loop
        show_loss = 'loss nan'
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            #logger.info("")
            for _ in trange(
                    steps_per_epoch,
                    desc=f"Epoch {epoch} {show_loss}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                output = self.model(**data)
                loss = output["loss"]*loss_decay if output['init_mean_icudays']-reinforcement_beta < output["mattching_mean_icudays"] else output["loss"]

                # backward

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(output["loss"].item())
                global_step += 1

                rf = True
                if rf:
                    # reinforcement learning
                    print (output['real_icudays'])

                    init_reward = output['real_icudays'] - reinforcement_beta
                    eachstep_reward = output["mattching_mean_icudays"]
                    counter = 0

                    while init_reward < eachstep_reward:

                        #print(f'RL: {init_reward} days -> {eachstep_reward} days')
                        loss_rl = torch.tensor(1+reinforcement_confidence*((eachstep_reward - init_reward) / (-eachstep_reward)), requires_grad=True)

                        loss_rl = torch.log(loss_rl)
                        loss_rl = torch.autograd.Variable(loss * reinforcement_alpha * (1-loss_decay) + (1 - reinforcement_alpha) * loss_rl, requires_grad=True)
                        loss_rl.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        init_reward = float(eachstep_reward)
                        output = self.model(**data)
                        loss = output["loss"]
                        eachstep_reward = output["mattching_mean_icudays"]
                        counter += 1
            show_loss = f'loss {sum(training_loss) / len(training_loss):.4f}'
            if demo_mode:
                self.save_ckpt(f"output/rl_epoch{epoch}.ckpt")

            if self.track_logging:
                # log and save
                logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
                logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
                if self.exp_path is not None:
                    self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None and epoch > val_start_epoch-1:
                scores = self.evaluate(val_dataloader)
                if self.track_logging:
                    logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                    for key in scores.keys():
                        logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        if self.track_logging:
                            logger.info(
                                f"New best {monitor} score ({score:.4f}) "
                                f"at epoch-{epoch}, step-{global_step}"
                            )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                self.early_stopping(score, monitor_criterion_ES=monitor_criterion)
            if self.early_stopping.early_stop:
                print ('Early Stop')
                break
        # load best model
        if load_best_model_at_last and self.exp_path is not None:
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))
        return

    def train_rl_b(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Optional[Dict[str, object]] = None,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
            reinforcement_confidence: float = 1,
            reinforcement_beta: float = 1,
            loss_decay : float = 0.5,
    ):
        """Trains the model.
        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        steps_per_epoch = len(train_dataloader)
        global_step = 0

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                    steps_per_epoch,
                    desc=f"Epoch {epoch} / {epochs}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                output = self.model(**data)
                loss = output["loss"]
                if epoch < 0:
                    # backward
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )

                elif epoch >= 0:
                    # reinforcement learning

                    init_reward = output['init_mean_icudays']
                    eachstep_reward = output["mattching_mean_icudays"]

                    if init_reward >= eachstep_reward:
                        loss_rl = torch.tensor(0.1 * ((eachstep_reward - init_reward) / (-eachstep_reward)),
                                               requires_grad=True)
                    else:
                        loss_rl = torch.tensor(10*((eachstep_reward - init_reward)/(-eachstep_reward)), requires_grad=True)
                        print(
                            f'RL: loss:{loss_rl}, {init_reward} days -> {eachstep_reward} days')
                    loss = torch.autograd.Variable(loss * 0.8 + loss_rl * 0.2, requires_grad=True)
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )

                training_loss.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                self.early_stopping(score, monitor_criterion_ES=monitor_criterion)
            if self.early_stopping.early_stop:
                print ('Early Stop')
                break
        # load best model
        if load_best_model_at_last and self.exp_path is not None:
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))
        return

    def train_rl_c(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Optional[Dict[str, object]] = None,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
    ):
        """Trains the model.
        Args:
            train_dataloader: Dataloader for training.
            val_dataloader: Dataloader for validation. Default is None.
            test_dataloader: Dataloader for testing. Default is None.
            epochs: Number of epochs. Default is 5.
            optimizer_class: Optimizer class. Default is torch.optim.Adam.
            optimizer_params: Parameters for the optimizer. Default is {"lr": 1e-3}.
            weight_decay: Weight decay. Default is 0.0.
            max_grad_norm: Maximum gradient norm. Default is None.
            monitor: Metric name to monitor. Default is None.
            monitor_criterion: Criterion to monitor. Default is "max".
            load_best_model_at_last: Whether to load the best model at the last.
                Default is True.
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        steps_per_epoch = len(train_dataloader)
        global_step = 0

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            # batch training loop
            logger.info("")
            for _ in trange(
                    steps_per_epoch,
                    desc=f"Epoch {epoch} / {epochs}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                # forward
                output = self.model(**data)
                loss = output["loss"]
                rf = True
                if rf:
                    # reinforcement learning

                    init_reward = output['init_mean_icudays']
                    eachstep_reward = output["mattching_mean_icudays"]

                    if init_reward < eachstep_reward:
                        print(
                            f'reinforment learning: original days was {init_reward}, learn from others with {eachstep_reward}days')
                        loss_rl = torch.tensor(1*(eachstep_reward - init_reward) / (-eachstep_reward), requires_grad=True)
                        loss = loss + loss_rl
                        #loss_rl.backward()
                        #optimizer.step()
                        #optimizer.zero_grad()
                        #init_reward = float(eachstep_reward)
                        #output = self.model(**data)
                        #eachstep_reward = output["mattching_mean_icudays"]


                # backward

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1



            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                self.early_stopping(score, monitor_criterion_ES=monitor_criterion)
            if self.early_stopping.early_stop:
                print ('Early Stop')
                break
        # load best model
        if load_best_model_at_last and self.exp_path is not None:
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            scores = self.evaluate(test_dataloader)
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))
        return

    def save_data(self, dataloader,
                  feature_list,
                  save_input_data=True,
                  save_output_data = False,
                  mode="test"):

        loss_all = []
        y_true_all = []
        y_prob_all = []
        input_data1, input_data2, input_data3, input_data4 = [],[],[],[]
        for data in tqdm(dataloader, desc="SAVE_DATA", disable=True):
            self.model.eval()
            with torch.no_grad():
                if save_input_data:
                    input_data1.extend(data[feature_list[0]])
                    input_data2.extend(data[feature_list[1]])
                    input_data3.extend(data[feature_list[2]])
                    input_data4.extend(data[feature_list[3]])

                if save_output_data:
                    output = self.model(**data)
                    #loss = output["loss"]
                    y_true = output["real_icudays2"]#.cpu().numpy()
                    y_prob = output["mattched_days"]#.cpu().numpy()
                    #loss_all.append(loss.item())

                    y_true_all.append(y_true)
                    y_prob_all.append(y_prob)

        if save_input_data:
            print(len(input_data1))
            # input_data = np.concatenate(input_data, axis=0)
            with open(f'output/labels/{mode}_input_{feature_list[0]}_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(input_data1, f)
            print(len(input_data2))
            # input_data = np.concatenate(input_data, axis=0)
            with open(f'output/labels/{mode}_input_{feature_list[1]}_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(input_data2, f)
            print(len(input_data3))
            # input_data = np.concatenate(input_data, axis=0)
            with open(f'output/labels/{mode}_input_{feature_list[2]}_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(input_data3, f)
            print(len(input_data4))
            # input_data = np.concatenate(input_data, axis=0)
            with open(f'output/labels/{mode}_input_{feature_list[3]}_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(input_data4, f)


        if save_output_data:
            y_true_all = np.concatenate(y_true_all, axis=0)
            y_prob_all = np.concatenate(y_prob_all, axis=0)
            with open(f'output/labels/True_epoch_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(y_true_all, f)
            with open(f'output/labels/Pred_epoch_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(y_prob_all, f)


    def inference(self, dataloader) -> Dict[str, float]:
        """Model inference.

        Args:
            dataloader: Dataloader for evaluation.

        Returns:
            y_true_all: List of true labels.
            y_prob_all: List of predicted probabilities.
            loss_mean: Mean loss over batches.
        """
        loss_all = []
        y_true_all = []
        y_prob_all = []
        real_icudays_all = []
        init_icudays_all = []
        mattching_icudays_all = []
        input_data = []
        for data in tqdm(dataloader, desc="Evaluation", disable=True):
            self.model.eval()
            with torch.no_grad():
                output = self.model(**data)
                input_data.extend(data["conditions"])
                loss = output["loss"]
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()

                loss_all.append(loss.item())
                real_icudays_all.append(output["real_icudays"])
                init_icudays_all.append(output["init_mean_icudays"])
                mattching_icudays_all.append(output["mattching_mean_icudays"])
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
        save_input_data = True
        if save_input_data == True:
            print (len(input_data))
            #input_data = np.concatenate(input_data, axis=0)
            with open(f'output/labels/True_input_conditions_{datetime.now().strftime("%H_%M_%S")}.json', 'wb') as f:
                pickle.dump(input_data, f)

        loss_mean = sum(loss_all) / len(loss_all)
        real_icudays_mean = sum(real_icudays_all) / len(real_icudays_all)
        init_icudays_mean = sum(init_icudays_all) / len(init_icudays_all)
        mattching_icudays_mean = sum(mattching_icudays_all) / len(mattching_icudays_all)
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_prob_all = np.concatenate(y_prob_all, axis=0)

        '''
        logger.info(
            f"init_icudays ({output['init_mean_icudays']:.4f}) "
        )
        logger.info(
            f"mattching_icudays ({output['mattching_mean_icudays']:.4f}) "
        )
        '''
        return y_true_all, y_prob_all, loss_mean, real_icudays_mean, init_icudays_mean, mattching_icudays_mean

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluates the model.

        Args:
            dataloader: Dataloader for evaluation.

        Returns:
            scores: a dictionary of scores.
        """
        y_true_all, y_prob_all, loss_mean, real_icudays_mean, init_icudays_mean, mattching_icudays_mean = self.inference(dataloader)

        mode = self.model.mode
        metrics_fn = get_metrics_fn(mode)
        scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics, dataset_name=self.dataset_name_for_metric) if mode == "multilabel" else metrics_fn(y_true_all, y_prob_all, metrics=self.metrics)

        scores["real_icudays"] = real_icudays_mean
        scores["init_icudays"] = init_icudays_mean
        scores["mattching_icudays"] = mattching_icudays_mean
        scores["loss"] = loss_mean
        return scores

    def save_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = self.model.state_dict()
        torch.save(state_dict, ckpt_path)
        return

    def load_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms
    from pyhealth.datasets.utils import collate_fn_dict

    class MNISTDataset(Dataset):
        def __init__(self, train=True):
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            self.dataset = datasets.MNIST(
                "../data", train=train, download=True, transform=transform
            )

        def __getitem__(self, index):
            x, y = self.dataset[index]
            return {"x": x, "y": y}

        def __len__(self):
            return len(self.dataset)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.mode = "multiclass"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x, y, **kwargs):
            x = torch.stack(x, dim=0).to(self.device)
            y = torch.tensor(y).to(self.device)
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            loss = self.loss(x, y)
            y_prob = torch.softmax(x, dim=1)
            return {"loss": loss, "y_prob": y_prob, "y_true": y}

    train_dataset = MNISTDataset(train=True)
    val_dataset = MNISTDataset(train=False)

    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn_dict, batch_size=64, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn_dict, batch_size=64, shuffle=False
    )

    model = Model()

    trainer = Trainer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        monitor="accuracy",
        epochs=5,
        test_dataloader=val_dataloader,
    )