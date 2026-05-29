EXPERIMENTS = {

    # =========================================================
    # BASELINE
    # =========================================================
    "baseline": {

        # =====================================================
        # MODEL
        # =====================================================
        "model": {
            "d_model": 64,
            "nhead": 4,
            "num_transformer_layers": 2,
            "dim_feedforward": 128,
            "classifier_hidden_dims": [],
            "pooling": "mean",
        },

        # =====================================================
        # TRAINING
        # =====================================================
        "training": {
            "batch_size": 64,
        },

        # =====================================================
        # OPTIMIZER
        # =====================================================
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },

        # =====================================================
        # LOSS
        # =====================================================
        "loss": {
            "name": "cross_entropy"
        },

    },



    "baseline_small": {

        # model kwargs
        "model": {
            "d_model": 64,
            "nhead": 4,
            "num_transformer_layers": 2,
            "dim_feedforward": 128,
            "classifier_hidden_dims": [],
            "pooling": "mean",
            "dropout": 0.1,
        },

        # training kwargs
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "epochs": 10,
            "weight_decay": 1e-4,
        }
    },

    # =========================================================
    # DEEPER TRANSFORMER
    # =========================================================
    "deep_transformer": {

        "model": {
            "d_model": 128,
            "nhead": 8,
            "num_transformer_layers": 4,
            "dim_feedforward": 256,
            "classifier_hidden_dims": [128],
            "pooling": "mean",
            "dropout": 0.1,
        },

        "training": {
            "batch_size": 16,
            "learning_rate": 5e-4,
            "epochs": 15,
            "weight_decay": 1e-4,
        }
    },

    # =========================================================
    # MLP ONLY
    # =========================================================
    "mlp_only": {

        "model": {
            "use_transformer": False,
            "d_model": 64,
            "classifier_hidden_dims": [128, 64],
            "pooling": "mean",
            "dropout": 0.2,
        },

        "training": {
            "batch_size": 64,
            "learning_rate": 1e-3,
            "epochs": 10,
            "weight_decay": 0.0,
        }
    },
}