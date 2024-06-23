from models.minGPT import GPT
from dataloaders.load_seq_data import LoadSeqData
from torch.distributed import init_process_group, destroy_process_group
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import timedelta
from utils import plot_losses
from tqdm import tqdm
import torch
import numpy as np
import os


class DNAGPTTrainer:
    def __init__(
        self,
        config,
    ):
        self.model_config = config.model_config
        self.optim_config = config.optim_config
        self.load_seq_data_config = config.load_seq_data_config
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.save_path = config.save_path
        self.master_addr = config.master_addr
        self.master_port = config.master_port
        self.backend = config.backend
        self.main_device = config.main_device
        self.process_timeout = config.process_timeout

        self.init_model = None
        self.best_model = self.initialize_model()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = torch.inf

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            self.save_path_losses = os.path.join(self.save_path, "losses")
            if not os.path.exists(self.save_path_losses):
                os.makedirs(self.save_path_losses)

            self.save_path_ckpt = os.path.join(self.save_path, "ckpt")
            if not os.path.exists(self.save_path_ckpt):
                os.makedirs(self.save_path_ckpt)

            self.save_path_config = os.path.join(self.save_path, "config")
            if not os.path.exists(self.save_path_config):
                os.makedirs(self.save_path_config)

            self.save_path_results = os.path.join(self.save_path, "results")
            if not os.path.exists(self.save_path_results):
                os.makedirs(self.save_path_results)

        self.save_config(config.__dict__)

    def save_config(
        self,
        config_dict,
    ):

        try:
            with open(os.path.join(self.save_path_config, "config.txt"), "w+") as f:
                for key, item in config_dict.items():
                    f.write(f"{key}: {item}\n")

        except Exception as e:
            print(str(e))

    def ddp_setup(self, rank, world_size):
        init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            timeout=timedelta(seconds=self.process_timeout),
        )

    def ddp_cleanup(self):
        destroy_process_group()

    def initialize_model(self):
        self.init_model = GPT(self.model_config)

        return self.init_model

    def lauch_ddp_model(self, device):
        model = GPT(self.model_config).to(device)
        model.load_state_dict(self.init_model.state_dict())

        return DDP(model, device_ids=[device])

    def save_ckpt(self, model, name):
        save_path = os.path.join(self.save_path_ckpt, name)
        torch.save(model.state_dict(), save_path)

    def initialize_epoch_losses(self):
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def update_global_losses(self):
        self.train_losses.append(np.mean(self.train_epoch_losses))
        self.val_losses.append(np.mean(self.val_epoch_losses))

    def update_epoch_losses(self, loss, type):
        if type == "train":
            self.train_epoch_losses.append(loss)

        else:
            self.val_epoch_losses.append(loss)

    def copy_models(self, models_from, models_to):
        for i in range(len(models_to)):
            if isinstance(models_to[i], list):
                for j in range(len(models_to[i])):
                    models_to[i][j].load_state_dict(models_from[i][j].state_dict())

            else:
                models_to[i].load_state_dict(models_from[i].state_dict())

    def multi_gpu_train(self, device, n_workers, loader):
        model = self.lauch_ddp_model(device)

        train_loader, val_loader = loader.get_train_loader(
            world_size=n_workers,
            rank=device,
        ), loader.get_val_loader(
            world_size=n_workers,
            rank=device,
        )

        optimizer = GPT.configure_optimizers(model, self.optim_config)

        if device == self.main_device:
            print("\n----- STARTING TRAINING -----\n")

        current_train_loss, current_val_loss = torch.inf, torch.inf

        for epoch in range(self.epochs):
            if device == self.main_device:
                self.initialize_epoch_losses()

            with tqdm(
                total=(len(train_loader) + len(val_loader)),
                desc=f"EPOCH {epoch+1}",
                disable=(device != self.main_device),
                postfix={
                    "train_loss": current_train_loss,
                    "val_loss": current_val_loss,
                },
            ) as bar:

                model.train()
                for train_data, train_labels in train_loader:

                    target = train_data.to(device)

                    train_data = loader.prepend_sos_token(train_data, crop_end=True).to(
                        device
                    )

                    logits, loss = model(train_data, target)

                    current_train_loss = loss.item()
                    if device == self.main_device:
                        bar.set_postfix(
                            {
                                "train_loss": current_train_loss,
                                "val_loss": current_val_loss,
                            }
                        )
                        self.update_epoch_losses(current_train_loss, type="train")

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.optim_config.grad_norm_clip
                    )
                    optimizer.step()

                    bar.update(n_workers)

                model.eval()
                with torch.no_grad():
                    for val_data, val_labels in val_loader:

                        target = val_data.to(device)

                        val_data = loader.prepend_sos_token(val_data, crop_end=True).to(
                            device
                        )

                        logits, loss = model(val_data, target)

                        current_val_loss = loss.item()
                        if device == self.main_device:
                            bar.set_postfix(
                                {
                                    "train_loss": current_train_loss,
                                    "val_loss": current_val_loss,
                                }
                            )
                            self.update_epoch_losses(current_val_loss, type="val")

                        bar.update(n_workers)

            dist.barrier()

            if device == self.main_device:
                self.update_global_losses()

                plot_losses(
                    self.train_losses,
                    self.val_losses,
                    save_path=os.path.join(self.save_path_losses, "losses.png"),
                )

                if self.val_losses[-1] < self.best_val_loss:
                    self.best_val_loss = self.val_losses[-1]

                    self.save_ckpt(model, "best.pt")

                    self.copy_models(
                        models_from=[model.module],
                        models_to=[self.best_model],
                    )

    def single_gpu_train(self, device):
        model = self.initialize_model().to(device)

        loader = LoadSeqData(self.load_seq_data_config)

        train_loader, val_loader = loader.get_train_loader(), loader.get_val_loader()

        optimizer = GPT.configure_optimizers(model, self.optim_config)

        print("\n----- STARTING TRAINING -----\n")

        current_train_loss, current_val_loss = torch.inf, torch.inf

        for epoch in range(self.epochs):
            self.initialize_epoch_losses()

            with tqdm(
                total=(len(train_loader) + len(val_loader)),
                desc=f"EPOCH {epoch+1}",
                postfix={
                    "train_loss": current_train_loss,
                    "val_loss": current_val_loss,
                },
            ) as bar:

                model.train()
                for train_data, train_labels in train_loader:

                    target = train_data.to(device)

                    train_data = loader.prepend_sos_token(train_data, crop_end=True).to(
                        device
                    )

                    logits, loss = model(train_data, target)

                    current_train_loss = loss.item()
                    bar.set_postfix(
                        {
                            "train_loss": current_train_loss,
                            "val_loss": current_val_loss,
                        }
                    )
                    self.update_epoch_losses(current_train_loss, type="train")

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.optim_config.grad_norm_clip
                    )
                    optimizer.step()

                    bar.update(1)

                model.eval()
                with torch.no_grad():
                    for val_data, val_labels in val_loader:

                        target = val_data.to(device)

                        val_data = loader.prepend_sos_token(val_data, crop_end=True).to(
                            device
                        )

                        logits, loss = model(val_data, target)

                        current_val_loss = loss.item()
                        bar.set_postfix(
                            {
                                "train_loss": current_train_loss,
                                "val_loss": current_val_loss,
                            }
                        )
                        self.update_epoch_losses(current_val_loss, type="val")

                        bar.update(1)

            self.update_global_losses()

            plot_losses(
                self.train_losses,
                self.val_losses,
                save_path=os.path.join(self.save_path_losses, "losses.png"),
            )

            if self.val_losses[-1] < self.best_val_loss:
                self.best_val_loss = self.val_losses[-1]

                self.save_ckpt(model, "best.ckpt")

                self.copy_models(
                    models_from=[model],
                    models_to=[self.best_model],
                )

    def evaluate_validation(self, device):
        loader = LoadSeqData(self.load_seq_data_config)

        val_loader = loader.get_val_loader()

        y_pred = []
        y_true = []

        with tqdm(
            total=len(val_loader),
            desc="EVALUATING MODEL ON VALIDATION SET",
        ) as bar:
            self.best_model = self.best_model.to(device)
            self.best_model.eval()
            with torch.no_grad():
                for val_data, _ in val_loader:

                    target = val_data.to(device)

                    val_data = loader.prepend_sos_token(val_data, crop_end=True).to(
                        device
                    )

                    logits, loss = self.best_model(val_data, target)

                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    predictions = torch.argmax(probs, dim=-1)

                    y_true.extend(target.cpu().numpy().flatten().tolist())
                    y_pred.extend(predictions.cpu().numpy().flatten().tolist())

                    bar.update(1)

        metrics_dict = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Score": [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, average="weighted", zero_division=0.0),
                recall_score(y_true, y_pred, average="weighted", zero_division=0.0),
                f1_score(y_true, y_pred, average="weighted", zero_division=0.0),
            ],
        }

        metrics_df = pd.DataFrame(metrics_dict)

        metrics_df.to_csv(
            os.path.join(self.save_path_results, "results.csv"),
            index=False,
        )

    def train_ddp_process(self, rank, world_size):
        self.ddp_setup(rank, world_size)

        loader = LoadSeqData(self.load_seq_data_config)

        self.multi_gpu_train(
            device=rank,
            n_workers=world_size,
            loader=loader,
        )

        self.ddp_cleanup()

    def spawn_train_ddp(self):
        self.initialize_model()

        world_size = torch.cuda.device_count()

        print(f"--> CUDA DEVICE COUNT: {world_size}")
        print(f"--> MASTER ADDR: {self.master_addr}")
        print(f"--> MASTER PORT: {self.master_port}")

        mp.spawn(
            self.train_ddp_process,
            args=(world_size,),
            nprocs=world_size,
        )
