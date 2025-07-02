# SPDX-License-Identifier: MIT
# © 2025 Manish Kumar

"""
    Filename: swiftqueue_trans.py

    Description:
    -----------
    Implements a Transformer-based model for predicting RTT and ECN decisions
    in a network queueing context, with a custom loss function that focuses on sharp changes in RTT.

    - SwiftQueue Transformer Model for RTT and ECN Prediction
    - Custom Loss Function for RTT Prediction
    - SwiftQueue Optimizer for Training the Transformer Model

    Contact:
    --------
    manish.kumar.iitd.cse@gmail.com
"""

import pandas as pd
import torch
import torch.nn as nn



""" SwiftQueue Transformer Model for RTT and ECN Prediction
This module implements a Transformer-based model for predicting Round Trip Time (RTT)
and Explicit Congestion Notification (ECN) decisions in a network queueing context.
- includes a custom loss function that focuses on sharp changes in RTT, which is crucial"""
class SwiftQueueTransformer(nn.Module):
    def __init__(self, feature_size=64, num_layers=2):
        super().__init__()
        # Transformer Encoder (Multi-head attention + feedforward) x2
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, 
            nhead=1, 
            dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output heads for predictions
        self.rtt_head = nn.Linear(feature_size, 1)                  # Predict RTT
        self.ecn_head = nn.Linear(feature_size, 1)                  # Predict ECN decision (L4S/classic)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.transformer(x)
        rtt_pred = self.rtt_head(x)                                 # shape: [batch, seq_len, 1]
        ecn_logits = self.ecn_head(x).squeeze(-1)                   # shape: [batch, seq_len]
        return rtt_pred, ecn_logits


""" Custom Loss Function for RTT Prediction"""
class ChangeFocusedLoss(nn.Module):
    def __init__(self, alpha=10, delta=20000, delta_percent=0.2):
        super().__init__()
        self.alpha = alpha
        self.delta = delta                                          # 20 ms in microseconds
        self.delta_percent = delta_percent

    def forward(self, pred, target):
        base_loss = torch.mean((pred - target)**2)
        rtt_diff = torch.abs(target[:, 1:] - target[:, :-1])
        rel_diff = rtt_diff / (target[:, :-1] + 1e-6)
        sharp_changes = (rtt_diff > self.delta) & (rel_diff > self.delta_percent)
        sharp_loss = torch.mean((pred[:, 1:][sharp_changes] - target[:, 1:][sharp_changes])**2)
        return base_loss + self.alpha * sharp_loss


""" SwiftQueue Optimizer for Training the Transformer Model"""
class SwiftQueueOptimizer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100
        )
        self.loss_fn_rtt = ChangeFocusedLoss()
        self.loss_fn_ecn = nn.BCEWithLogitsLoss()

    def partial_fit(self, batch):
        self.model.train()
        inputs, (rtt_targets, ecn_labels) = batch
        self.optimizer.zero_grad()
        rtt_preds, ecn_logits = self.model(inputs)

        loss_rtt = self.loss_fn_rtt(rtt_preds.squeeze(-1), rtt_targets)
        loss_ecn = self.loss_fn_ecn(ecn_logits, ecn_labels)
        loss = loss_rtt + loss_ecn

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "rtt_loss": loss_rtt.item(),
            "ecn_loss": loss_ecn.item()
        }