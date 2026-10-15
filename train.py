"""CLI entrypoint: python train.py --config configs/custom_cnn.yaml"""
import argparse

import mlflow
import numpy as np
import yaml
from tensorflow import keras

from src.data import fashion_mnist, flowers
from src.data.fashion_mnist import normalization_stats
from src.models import custom_cnn, resnet34, transfer


def _build_optimizer(cfg: dict):
    kind = cfg.get("type", "adam").lower()
    if kind == "sgd":
        return keras.optimizers.SGD(
            learning_rate=cfg.get("learning_rate", 0.01),
            momentum=cfg.get("momentum", 0.0),
            nesterov=cfg.get("nesterov", False),
            decay=cfg.get("decay", 0.0),
        )
    if kind == "nadam":
        return keras.optimizers.Nadam(learning_rate=cfg.get("learning_rate", 1e-3))
    if kind == "adam":
        return keras.optimizers.Adam(learning_rate=cfg.get("learning_rate", 1e-3))
    raise ValueError(f"Unknown optimizer type: {kind!r}")


def _flatten_params(cfg: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict into dot-separated keys for MLflow."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=key))
        else:
            out[key] = v
    return out


class _MLflowCallback(keras.callbacks.Callback):
    """Logs per-epoch train/val metrics to the active MLflow run."""

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(logs, step=epoch)


def train_custom_cnn(cfg: dict):
    mp = cfg["model_params"]
    tc = cfg["training"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = fashion_mnist.load()

    model = custom_cnn.build(
        input_shape=tuple(mp["input_shape"]),
        n_classes=mp["n_classes"],
        dropout=mp.get("dropout", 0.5),
    )
    model.compile(loss=tc["loss"], optimizer=tc.get("optimizer", "nadam"), metrics=tc["metrics"])
    history = model.fit(
        X_train, y_train,
        epochs=tc["epochs"],
        validation_data=(X_valid, y_valid),
        callbacks=[_MLflowCallback()],
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metrics({"test_accuracy": acc, "test_loss": loss})
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    return model, history


def train_resnet34(cfg: dict):
    mp = cfg["model_params"]
    tc = cfg["training"]
    resize = tuple(mp.get("resize_to", [224, 224]))

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = fashion_mnist.load(resize_to=resize)

    model = resnet34.build(input_shape=tuple(mp["input_shape"]), n_classes=mp["n_classes"])
    model.compile(loss=tc["loss"], optimizer=tc.get("optimizer", "adam"), metrics=tc["metrics"])
    history = model.fit(
        X_train, y_train,
        epochs=tc["epochs"],
        batch_size=tc.get("batch_size", 32),
        validation_data=(X_valid, y_valid),
        callbacks=[_MLflowCallback()],
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metrics({"test_accuracy": acc, "test_loss": loss})
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    return model, history


def train_xception(cfg: dict):
    tc = cfg["training"]
    train_set, valid_set, _, n_classes, dataset_size, _ = flowers.load(batch_size=tc["batch_size"])

    model, base_model = transfer.build(n_classes=n_classes, base=cfg.get("base", "xception"))

    # Phase 1: train only the new head
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        loss=tc["loss"],
        optimizer=_build_optimizer(tc["freeze_optimizer"]),
        metrics=tc["metrics"],
    )
    model.fit(
        train_set,
        steps_per_epoch=int(0.75 * dataset_size / tc["batch_size"]),
        validation_data=valid_set,
        validation_steps=int(0.15 * dataset_size / tc["batch_size"]),
        epochs=tc["freeze_epochs"],
        callbacks=[_MLflowCallback()],
    )

    # Phase 2: unfreeze and fine-tune the whole network
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(
        loss=tc["loss"],
        optimizer=_build_optimizer(tc["finetune_optimizer"]),
        metrics=tc["metrics"],
    )
    history = model.fit(
        train_set,
        steps_per_epoch=int(0.75 * dataset_size / tc["batch_size"]),
        validation_data=valid_set,
        validation_steps=int(0.15 * dataset_size / tc["batch_size"]),
        epochs=tc["finetune_epochs"],
        callbacks=[_MLflowCallback()],
    )
    return model, history


_TRAINERS = {
    "custom_cnn": train_custom_cnn,
    "resnet34": train_resnet34,
    "xception": train_xception,
}


def main():
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--save", default=None, help="Path to save the trained model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]
    if model_name not in _TRAINERS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(_TRAINERS)}")

    mlflow.set_experiment(model_name)
    with mlflow.start_run():
        mlflow.log_params(_flatten_params(cfg))

        print(f"Training {model_name} ...")
        model, _ = _TRAINERS[model_name](cfg)

        if args.save:
            save_path = args.save if args.save.endswith(".keras") else args.save + ".keras"
            model.save(save_path)
            mlflow.log_artifact(save_path)
            print(f"Saved to {save_path}")

            # Save per-pixel normalization stats for fashion_mnist models so
            # inference code can apply the same transform used during training.
            if cfg.get("dataset") == "fashion_mnist" and cfg["model"] != "resnet34":
                stats_path = save_path.replace(".keras", "_norm_stats.npz")
                X_mean, X_std = normalization_stats()
                np.savez(stats_path, X_mean=X_mean, X_std=X_std)
                mlflow.log_artifact(stats_path)
                print(f"Saved normalization stats to {stats_path}")


if __name__ == "__main__":
    main()
