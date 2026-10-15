# CNN Models for Image Classification

Convolutional neural network implementations in TensorFlow/Keras, structured as a runnable Python package with a CLI training interface, YAML configs, MLflow experiment tracking, and evaluation tooling. Trained models are served two ways: a Dockerized FastAPI REST API and a Gradio web demo.

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/apayne1850/cv-cnns)**

## Models

| Model | Dataset | Architecture | Params | Test Accuracy | Inference (CPU) | Notes |
|---|---|---|---|---|---|---|
| `custom_cnn` | Fashion-MNIST | Conv2D → MaxPool → Dense | 1.4M | **88.4%** | 174ms | Trained from scratch, 3 epochs on CPU |
| `resnet34` | Fashion-MNIST (upscaled to 224×224) | 34-layer residual network | 21.3M | pending | 259ms | Custom `ResidualUnit` layer |
| `xception` | tf_flowers | Xception (ImageNet pretrained) | 20.9M | pending | 369ms | Two-phase transfer learning |

> **The datasets differ, so the accuracies can't all be compared.** `custom_cnn` and `resnet34` are both trained on Fashion-MNIST (10 classes, grayscale), so you can compare those two. `xception` is trained on tf_flowers (5 classes, RGB photos), so its accuracy can't be compared with the other two.
>
> ResNet-34 and Xception accuracy is pending. Those runs are set up for the SLURM GPU cluster (see [Training on a SLURM GPU cluster](#training-on-a-slurm-gpu-cluster)), and the table will be updated when they finish.

## Project structure

```
src/
  models/         # model definitions (custom_cnn, resnet34, transfer)
  data/           # dataset loaders (fashion_mnist, flowers)
  utils/          # preprocessing and metrics helpers
configs/          # YAML hyperparameter configs per model
notebooks/        # original exploration notebooks
tests/            # pytest unit tests
train.py          # CLI training entrypoint
evaluate.py       # evaluation + confusion matrix
predict.py        # single-image inference
api.py            # FastAPI inference server
demo.py           # Gradio web demo
```

## Setup

> Tested on Python 3.10, which is the version used by CI, the Docker image and the SLURM cluster environment. Newer TensorFlow releases also support 3.11+, but this project hasn't been tested on them.

```bash
pip install -r requirements-dev.txt
```

## Training

```bash
# Train the custom CNN
python train.py --config configs/custom_cnn.yaml --save saved_models/custom_cnn

# Train ResNet-34 from scratch
python train.py --config configs/resnet34.yaml --save saved_models/resnet34

# Transfer learning with Xception on tf_flowers
python train.py --config configs/xception.yaml --save saved_models/xception

# Or via Make
make train-cnn
```

Edit any `configs/*.yaml` file to change epochs, optimizer, learning rate, etc. without touching the code.

### Training on a SLURM GPU cluster

`run.sh` is a SLURM batch script that trains and evaluates ResNet-34 and Xception on a single GPU:

```bash
sbatch run.sh
# logs go to results/slurm/cnn-train_<jobid>.log
```

**Status:** I ran this job on the university's SLURM GPU cluster and debugged it there. The fixes were:

- pointing `LD_LIBRARY_PATH` at the pip-installed CUDA libraries
- raising the memory request to 64G after out-of-memory failures
- moving the 224×224 upscale of Fashion-MNIST for ResNet-34 to the CPU, because the GPU `ResizeBilinear` kernel hits an int32 overflow at about 3 billion elements

The full benchmark runs haven't finished yet, so the ResNet-34 and Xception accuracies above are still pending.

## Experiment tracking (MLflow)

Every `train.py` run is logged to MLflow. Each model gets its own experiment (`custom_cnn`, `resnet34`, `xception`). For each run, MLflow records:

- **Parameters:** the whole YAML config, flattened into keys such as `training.epochs` and `finetune_optimizer.learning_rate`
- **Metrics:** `loss`, `accuracy`, `val_loss` and `val_accuracy` for every epoch. The `custom_cnn` and `resnet34` runs also log `test_accuracy` and `test_loss`.
- **Artifacts:** the saved `.keras` model when you pass `--save`. For `custom_cnn`, the normalization stats file (`*_norm_stats.npz`) used at inference time is logged too.

Runs are stored locally (`mlruns/` or `mlflow.db`, both git-ignored). To browse and compare them:

```bash
make ui
# or: mlflow ui
```

Then open http://localhost:5000.

## Evaluation

```bash
python evaluate.py --model saved_models/custom_cnn.keras --confusion-matrix
```

Prints per-class precision/recall/F1 and optionally renders a confusion matrix.

## Inference

```bash
python predict.py --model saved_models/custom_cnn.keras --image my_image.png
```

## Gradio Demo

Run a local interactive demo in your browser:

```bash
make demo
# or: python demo.py
```

Upload any clothing image and the model returns top-5 class predictions with confidence scores.

**Deploy to Hugging Face Spaces** (free, gives you a public URL):
1. Create a new Space at huggingface.co → Gradio SDK
2. Push `demo.py`, `src/`, and `requirements.txt` to the Space repo
3. Upload your trained model: `git lfs track "*.keras"` then add `saved_models/custom_cnn.keras`

## API (FastAPI)

Start the inference server locally:

```bash
make serve
# or: uvicorn api:app --reload
```

`POST /predict` — upload an image, get back a class label and confidence scores:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_image.png"
# {"class":"Sneaker","confidence":0.9821,"scores":{...}}
```

`GET /health` — liveness check.

## Docker

```bash
# Build and run
make docker-up

# Equivalent to:
docker compose up
```

This runs the FastAPI service in Docker. The image contains only the code and dependencies, not the model. The trained model is mounted from `./saved_models`, so after retraining you just restart the container and don't need to rebuild the image. To switch models, set `MODEL_PATH` and `MODEL_TYPE` in `docker-compose.yml`.

## Tests

```bash
pytest tests/ -v
# or
make test
```

CI runs automatically on every push via GitHub Actions.

## Requirements

- tensorflow >= 2.12
- tensorflow-datasets
- scikit-learn, matplotlib, seaborn
- fastapi, uvicorn, python-multipart
- pyyaml, pillow
- mlflow
