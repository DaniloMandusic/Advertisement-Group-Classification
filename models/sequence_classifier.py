import torch
import torch.nn as nn


import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            max_seq_len,
            num_classes,

            # embedding
            d_model=64,
            embedding_dropout=0.1,

            # transformer
            use_transformer=True,
            nhead=4,
            num_transformer_layers=2,
            dim_feedforward=128,
            transformer_dropout=0.1,

            # extra dense layers after pooling
            classifier_hidden_dims=None,   # e.g. [128, 64]
            classifier_dropout=0.1,

            # pooling strategy
            pooling="mean",  # "mean", "max", "cls"

            # normalization
            use_layernorm=False,
    ):
        super().__init__()

        if classifier_hidden_dims is None:
            classifier_hidden_dims = []

        self.pooling = pooling
        self.use_transformer = use_transformer

        # =========================================================
        # EMBEDDINGS
        # =========================================================
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # optional CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # optional normalization
        self.layernorm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # =========================================================
        # TRANSFORMER STACK
        # =========================================================
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
                batch_first=True,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers
            )
        else:
            self.encoder = nn.Identity()

        # =========================================================
        # CLASSIFIER STACK
        # =========================================================
        classifier_layers = []

        input_dim = d_model

        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
            ])
            input_dim = hidden_dim

        classifier_layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, input_ids, attention_mask):
        B, S = input_ids.shape

        # =========================================================
        # TOKEN + POSITION EMBEDDINGS
        # =========================================================
        positions = torch.arange(
            S,
            device=input_ids.device
        ).unsqueeze(0)

        x = self.embedding(input_ids)
        x = x + self.pos_embedding(positions)

        # =========================================================
        # OPTIONAL CLS TOKEN
        # =========================================================
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)

            x = torch.cat([cls_tokens, x], dim=1)

            cls_mask = torch.ones(
                (B, 1),
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )

            attention_mask = torch.cat(
                [cls_mask, attention_mask],
                dim=1
            )

        x = self.embedding_dropout(x)
        x = self.layernorm(x)

        # =========================================================
        # TRANSFORMER
        # =========================================================
        src_key_padding_mask = (attention_mask == 0)

        x = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        # =========================================================
        # POOLING
        # =========================================================
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()

            x = (
                    (x * mask_expanded).sum(dim=1)
                    / mask_expanded.sum(dim=1).clamp(min=1e-9)
            )

        elif self.pooling == "max":
            mask_expanded = attention_mask.unsqueeze(-1)

            x = x.masked_fill(mask_expanded == 0, -1e9)
            x = x.max(dim=1).values

        elif self.pooling == "cls":
            x = x[:, 0]

        else:
            raise ValueError(
                f"Unknown pooling method: {self.pooling}"
            )

        # =========================================================
        # CLASSIFIER
        # =========================================================
        logits = self.classifier(x)

        return logits