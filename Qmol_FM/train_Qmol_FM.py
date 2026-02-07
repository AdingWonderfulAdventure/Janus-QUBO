# train_Qmol_FM.py

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# ==============================================================================
# 1. 极简 DataModule (仅用于包装已加载的数据集)
# ==============================================================================
log = logging.getLogger(__name__)
class PreloadedDataModule(pl.LightningDataModule):
    """A minimal DataModule for wrapping datasets already loaded into memory."""
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, 
                 batch_size: int, num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4
        )

# ==============================================================================
# 2. 模型定义 (TorchFM 和 FMSurrogate)
# ==============================================================================

class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k))
        self.lin = nn.Linear(n, 1, bias=True)
        nn.init.normal_(self.V, mean=0, std=0.01)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(x, self.V.pow(2)).sum(1, keepdim=True)
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        return out.squeeze(dim=1)

class FMSurrogate(pl.LightningModule):
    def __init__(self, n_features, factor_k, learning_rate, weight_decay, loss_type="mse",**kwargs):
        super().__init__()
        self.save_hyperparameters('n_features', 'factor_k', 'learning_rate', 'weight_decay', 'loss_type')
        self.model = TorchFM(n=n_features, k=factor_k)
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss(beta=1.0)  # beta=delta，可以调
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32)
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32)
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        output = {
            'preds': y_hat.detach().clone(),
            'targets': y.detach().clone()
        }
        self.validation_step_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        outputs = self.validation_step_outputs
        preds = torch.cat([x['preds'] for x in outputs]).float().cpu().numpy()
        targets = torch.cat([x['targets'] for x in outputs]).float().cpu().numpy()
        
        r_squared, mae, rmse = 0.0, 0.0, 0.0
        mask = np.isfinite(preds) & np.isfinite(targets)
        if np.sum(mask) > 1:
            try:
                # --- R² ---
                r_squared = r2_score(targets[mask], preds[mask])
                # --- MAE ---
                mae = mean_absolute_error(targets[mask], preds[mask])
                
                # --- RMSE ---
                rmse = math.sqrt(mean_squared_error(targets[mask], preds[mask]))
            except Exception as e:
                print(f"Warning: Could not compute r2_score. Error: {e}")
                r_squared = 0.0
        
        # ✅ 记录到日志
        self.log('val_r2', r_squared, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        def lr_lambda(current_step: int):
            if self.num_training_steps == 0: return 1.0 # Avoid division by zero
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit':
            # This logic correctly calculates training steps based on the dataloader
            train_loader = self.trainer.datamodule.train_dataloader()
            
            # Handles limit_train_batches correctly whether it's int, float, or 1.0
            if hasattr(self.trainer, 'limit_train_batches') and self.trainer.limit_train_batches is not None:
                limit_batches = self.trainer.limit_train_batches
                if limit_batches == 0.0:
                    num_batches = 0
                elif isinstance(limit_batches, int):
                    num_batches = min(limit_batches, len(train_loader))
                else: # float
                    num_batches = int(len(train_loader) * limit_batches)
            else:
                 num_batches = len(train_loader)

            effective_max_epochs = self.trainer.max_epochs or 1
            self.num_training_steps = num_batches * effective_max_epochs
            self.num_warmup_steps = int(self.num_training_steps * 0.05)
            self.trainer.print(f"Total training steps: {self.num_training_steps}, Warmup steps: {self.num_warmup_steps}")


class MLP(nn.Module):
    """
    A standard, flexible and robust Multi-Layer Perceptron (MLP) model.
    Designed for tabular data (e.g. latent codes).
    """
    def __init__(
        self, 
        n_features: int, 
        hidden_dims: list = None, 
        dropout: float = 0.3,
    ):
        """
        Initialize MLP model.

        Args:
            n_features (int): 输入特征的数量 (例如，latent_code 的维度，128)。
            hidden_dims (list, optional): 一个整数列表，定义了每个隐藏层的神经元数量。
                                          例如 [512, 256, 128]。
                                          默认为 [256, 128]。
            dropout (float, optional): 在每个隐藏层之后应用的 Dropout 比率。
                                       默认为 0.3。
        """
        super().__init__()

        # Use reasonable defaults if hidden_dims not provided
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        layers = []
        input_dim = n_features
        
        # Dynamically build hidden layers
        for h_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(input_dim, h_dim))
            
            # Batch normalization
            # Improves training speed and stability
            layers.append(nn.BatchNorm1d(h_dim))
            
            # Activation function
            # ReLU is the most common and efficient choice
            layers.append(nn.ReLU())
            
            # Dropout (regularization)
            # 在训练时随机“关闭”一些神经元，防止过拟合
            layers.append(nn.Dropout(dropout))
            
            # Update input dimension for next layer
            input_dim = h_dim
        
        # Add final output layer
        # Output a single scalar value (predicted score)
        layers.append(nn.Linear(input_dim, 1))
        
        # Pack all layers into a network using nn.Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): 输入的批次数据，形状为 (batch_size, n_features)。

        Returns:
            torch.Tensor: 模型的输出，形状为 (batch_size,)。
        """
        # squeeze(-1) 将输出的形状从 (batch_size, 1) 变为 (batch_size)
        return self.net(x).squeeze(-1)
    
class MLPSurrogate(pl.LightningModule):
    def __init__(self, n_features, hidden_dims, dropout, learning_rate, weight_decay, loss_type="mse", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate MLP model
        self.model = MLP(
            n_features=n_features, 
            hidden_dims=hidden_dims, 
            dropout=dropout
        )
        
        # (Optional) Use torch.compile
        
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss(beta=1.0)  # beta=delta，可以调
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Custom prediction step.
        DataLoader returns batch as a list [tensor] for prediction, we need to unpack it.
        """
        # 1. Unpack the actual input tensor from the list
        if isinstance(batch, (list, tuple)):
             x = batch[0]
        else: # If batch is already a tensor, use directly
             x = batch

        # 2. Call model forward with unpacked tensor
        return self(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32)
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32)
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        output = {
            'preds': y_hat.detach().clone(),
            'targets': y.detach().clone()
        }
        self.validation_step_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        outputs = self.validation_step_outputs
        
        # ✅ 关键修复：在调用 .numpy() 之前，使用 .float() 将数据类型转换为 float32
        preds_list = [x['preds'] for x in outputs]
        preds = torch.cat(preds_list).float().cpu().numpy()

        # Apply same conversion to targets for consistency
        targets_list = [x['targets'] for x in outputs]
        targets = torch.cat(targets_list).float().cpu().numpy()
        
        r_squared, mae, rmse = 0.0, 0.0, 0.0
        mask = np.isfinite(preds) & np.isfinite(targets)
        if np.sum(mask) > 1:
            try:
                # --- R² ---
                r_squared = r2_score(targets[mask], preds[mask])
                # --- MAE ---
                mae = mean_absolute_error(targets[mask], preds[mask])
                
                # --- RMSE ---
                rmse = math.sqrt(mean_squared_error(targets[mask], preds[mask]))
            except Exception as e:
                print(f"Warning: Could not compute r2_score. Error: {e}")
                r_squared = 0.0
        
        # ✅ 记录到日志
        self.log('val_r2', r_squared, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        def lr_lambda(current_step: int):
            if self.num_training_steps == 0: return 1.0 # Avoid division by zero
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit':
            # This logic correctly calculates training steps based on the dataloader
            train_loader = self.trainer.datamodule.train_dataloader()
            
            # Handles limit_train_batches correctly whether it's int, float, or 1.0
            if hasattr(self.trainer, 'limit_train_batches') and self.trainer.limit_train_batches is not None:
                limit_batches = self.trainer.limit_train_batches
                if limit_batches == 0.0:
                    num_batches = 0
                elif isinstance(limit_batches, int):
                    num_batches = min(limit_batches, len(train_loader))
                else: # float
                    num_batches = int(len(train_loader) * limit_batches)
            else:
                num_batches = len(train_loader)

            effective_max_epochs = self.trainer.max_epochs or 1
            self.num_training_steps = num_batches * effective_max_epochs
            self.num_warmup_steps = int(self.num_training_steps * 0.05)
            self.trainer.print(f"Total training steps: {self.num_training_steps}, Warmup steps: {self.num_warmup_steps}")

class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ffn: int,
        dropout: float = 0.1,
        shared_embedding: bool = True,  # Whether to share embedding
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # --- Embedding layer ---
        n_tokens = 3  # {0, 1, CLS}
        if shared_embedding:
            self.feature_embeddings = nn.Embedding(n_tokens, d_model)
        else:
            self.feature_embeddings = nn.ModuleList([
                nn.Embedding(n_tokens, d_model) for _ in range(n_features)
            ])
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=F.relu,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.shared_embedding = shared_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, n_features], dtype=float32 (0/1)
        x_long = x.long()
        
        # --- Feature tokenization ---
        if self.shared_embedding:
            feature_embeds = self.feature_embeddings(x_long)  # [B, F, d_model]
        else:
            embeds = [emb(x_long[:, i]) for i, emb in enumerate(self.feature_embeddings)]
            feature_embeds = torch.stack(embeds, dim=1)  # [B, F, d_model]
        
        # CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        
        transformer_input = torch.cat([cls_tokens, feature_embeds], dim=1)  # [B, F+1, d_model]
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Extract CLS output
        cls_output = transformer_output[:, 0, :]
        
        prediction = self.prediction_head(cls_output)
        return prediction.squeeze(-1)

class FTTransformerSurrogate(pl.LightningModule):
    def __init__(self, n_features, d_model, n_head, n_layers, d_ffn,
                 dropout, learning_rate, weight_decay, loss_type="mse", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = FTTransformer(
            n_features=n_features,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            shared_embedding=True
        )
        
        # Loss function selection
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32).to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_step", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_epoch", loss, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32).to(self.device)
        y = y.to(self.device)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({"preds": y_hat.detach(), "targets": y.detach()})
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        outputs = self.validation_step_outputs
        
        # ✅ 关键修复：在调用 .numpy() 之前，使用 .float() 将数据类型转换为 float32
        preds_list = [x['preds'] for x in outputs]
        preds = torch.cat(preds_list).float().cpu().numpy()

        # Apply same conversion to targets for consistency
        targets_list = [x['targets'] for x in outputs]
        targets = torch.cat(targets_list).float().cpu().numpy()
        
        r_squared, mae, rmse = 0.0, 0.0, 0.0
        mask = np.isfinite(preds) & np.isfinite(targets)
        if np.sum(mask) > 1:
            try:
                # --- R² ---
                r_squared = r2_score(targets[mask], preds[mask])
                # --- MAE ---
                mae = mean_absolute_error(targets[mask], preds[mask])
                
                # --- RMSE ---
                rmse = math.sqrt(mean_squared_error(targets[mask], preds[mask]))
            except Exception as e:
                print(f"Warning: Could not compute r2_score. Error: {e}")
                r_squared = 0.0
        
        # ✅ 记录到日志
        self.log('val_r2', r_squared, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        def lr_lambda(current_step: int):
            if self.num_training_steps == 0: return 1.0 # Avoid division by zero
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit':
            # This logic correctly calculates training steps based on the dataloader
            train_loader = self.trainer.datamodule.train_dataloader()
            
            # Handles limit_train_batches correctly whether it's int, float, or 1.0
            if hasattr(self.trainer, 'limit_train_batches') and self.trainer.limit_train_batches is not None:
                limit_batches = self.trainer.limit_train_batches
                if limit_batches == 0.0:
                    num_batches = 0
                elif isinstance(limit_batches, int):
                    num_batches = min(limit_batches, len(train_loader))
                else: # float
                    num_batches = int(len(train_loader) * limit_batches)
            else:
                num_batches = len(train_loader)

            effective_max_epochs = self.trainer.max_epochs or 1
            self.num_training_steps = num_batches * effective_max_epochs
            self.num_warmup_steps = int(self.num_training_steps * 0.05)
            self.trainer.print(f"Total training steps: {self.num_training_steps}, Warmup steps: {self.num_warmup_steps}")


# ==============================================================================
# 3. 主训练函数 (训练引擎的入口)
# ==============================================================================
def main_train(args: argparse.Namespace, train_dataset: Dataset, val_dataset: Dataset):
    # Use logger controlled by main script
    log = logging.getLogger(__name__)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')
    
    dm = PreloadedDataModule(
        train_dataset=train_dataset, val_dataset=val_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.model_type == 'fm':
        model = FMSurrogate(
            n_features=args.n_features, factor_k=args.factor_k,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            loss_type=args.loss_type
        )
    elif args.model_type == 'mlp':
        model = MLPSurrogate(
            n_features=args.n_features, hidden_dims=args.mlp_hidden_dims,
            dropout=args.mlp_dropout, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,loss_type=args.loss_type
        )

    elif args.model_type == 'ftt':
        model = FTTransformerSurrogate(
        n_features=args.n_features,
        d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers, d_ffn=args.d_ffn,
        dropout=args.ft_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    log_hparams = vars(args).copy()
    experiment_name = args.swanlab_experiment_name or f"fm_k{args.factor_k}_lr{args.learning_rate}"
    
    swanlab_logger = SwanLabLogger(
        project=args.swanlab_project, experiment_name=experiment_name,
        save_dir=args.save_dir, log_hyperparams=log_hparams
    )
    checkpoint_dir = Path(args.save_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, monitor='val_loss', mode='min', save_top_k=3,
        filename='best-model-{epoch:02d}-{val_loss:.4f}-{val_r2:.4f}'
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min',min_delta=1e-3)
    callbacks = [checkpoint_callback, early_stopping_callback, LearningRateMonitor(logging_interval='epoch')]

    strategy = DDPStrategy(find_unused_parameters=False, start_method="fork") if len(args.devices) > 1 else "auto"

    trainer = pl.Trainer(
        accelerator="gpu", devices=args.devices, max_epochs=args.max_epochs,
        callbacks=callbacks, logger=swanlab_logger, precision="bf16-mixed", 
        strategy=strategy,
        num_sanity_val_steps=0, log_every_n_steps=10,
        gradient_clip_val=1.0
    )

    log.info(f"Starting surrogate model training...")
    # Return Trainer and related objects instead of calling .fit()
    return trainer, model, dm, checkpoint_callback

if __name__ == '__main__':
    print("This script is a library for training the surrogate model and is not meant to be run directly.")
    print("Please run 'active_learning_loop.py' to start the process.")