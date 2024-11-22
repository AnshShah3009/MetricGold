# MetricGold: Repurposing Diffusion-Based Image Generators for Monocular Metric Depth Estimation
[[Arxiv]]([https://github.com/AnshShah3009/MetricGold/blob/main/MetricGold.pdf](https://arxiv.org/abs/2411.10886)), The paper needs a lot of updates to complete, but the root idea is expressed.

We build upon the Marigold pipeline which predicts Monocular Relative Depth and train that pipeline for Log Scaled Monocular Metric Depth taking inspiration from DMD(Diffusion Models for Metric Depth). The idea is that we augment the metric depth in such a fashion that the VAE reconstruction is good and there's little need to finetune the VAE. Now we can leverage the LDM pipeline to finetune a model which predicts Metric Depth. All you need to do is a inverse of a simple function. We also aligned with the hypothesis of the Depth Anything V2 paper which states that Photorealistic Synthetic Data is better as Real data has sensor noise even when it has been preprocessed. Simulators can help us have a good initilisation and data redistillation like the SAM paper can improve the performance of our model due to domain expansion caused by distillation. We did not have the resources to get to the second step of the project where we pseudo label and retrain the model. But the idea was the same.

function used for Augmentation : D_aug = log(D / D_min) / log(D_max / D_min)
Inverse to get back the metric depth : D = e^(D_aug,

# TODO Update Repo Installation

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/AnshShah3009/MetricGold.git
cd Marigold
```

### 💻 Dependencies

We provide several ways to install the dependencies.

1. **Using [Mamba](https://github.com/mamba-org/mamba)**, which can installed together with [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3). 

    Windows users: Install the Linux version into the WSL.

    After the installation, Miniforge needs to be activated first: `source /home/$USER/miniforge3/bin/activate`.

    Create the environment and install dependencies into it:

    ```bash
    mamba env create -n marigold --file environment.yaml
    conda activate marigold
    ```

2. **Using pip:** 
    Alternatively, create a Python native virtual environment and install dependencies into it:

    ```bash
    python -m venv venv/metricgold
    source venv/metricgold/bin/activate
    pip install -r requirements.txt
    ```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing on your images

### 📷 Prepare images

1. Use selected images from our paper:

    ```bash
    bash script/download_sample_data.sh
    ```

1. Or place your images in a directory, for example, under `input/in-the-wild_example`, and run the following inference command.

```bash
 python run.py \
     --input_rgb_dir input/in-the-wild_example \
     --output_dir output/in-the-wild_example_lcm
 ```

### 🎮 Run inference with DDIM (paper setting)

This setting corresponds to our paper. For academic comparison, please run with this setting.

```bash
python run.py \
    --checkpoint prs-eth/Train_MetricGold \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

You can find all results in `output/in-the-wild_example`. Enjoy!

### ⚙️ Inference settings

The default settings are optimized for the best result. However, the behavior of the code can be customized:

- Trade-offs between the **accuracy** and **speed** (for both options, larger values result in better accuracy at the cost of slower inference.)
  - `--ensemble_size`: Number of inference passes in the ensemble. For LCM `ensemble_size` is more important than `denoise_steps`. Default: ~~10~~ 5 (for LCM).
  - `--denoise_steps`: Number of denoising steps of each inference pass. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps. When unassigned (`None`), will read default setting from model config. Default: ~~10 4 (for LCM)~~ `None`.

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  
  
  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: ~~768~~ `None`.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to have faster speed and reduced VRAM usage, but might lead to suboptimal results.
- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing `--batch_size 1` helps to increase reproducibility. To ensure full reproducibility, [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) needs to be used.
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.

### ⬇ Checkpoint cache

By default, the [checkpoint](https://huggingface.co/prs-eth/marigold-v1-0) is stored in the Hugging Face cache.
The `HF_HOME` environment variable defines its location and can be overridden, e.g.:

```bash
export HF_HOME=$(pwd)/cache
```

At inference, specify the checkpoint path:

```bash
python run.py \
    --checkpoint checkpoint/marigold-v1-0 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example\
    --output_dir output/in-the-wild_example
```

## 🦿 Evaluation on test datasets <a name="evaluation"></a>

Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
```
```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```

Run inference and evaluation scripts, for example:

```bash
# Run inference
bash script/eval/11_infer_nyu.sh

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

Note: although the seed has been set, the results might still be slightly different on different hardware.

## 🏋️ Training

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR  # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

Run training script

```bash
python train.py --config config/train_marigold.yaml
```

Resume from a checkpoint, e.g.

```bash
python train.py --resume_run output/marigold_base/checkpoint/latest
```

Evaluating results

Only the U-Net is updated and saved during training. To use the inference pipeline with your training result, replace `unet` folder in Marigold checkpoints with that in the `checkpoint` output folder. Then refer to [this section](#evaluation) for evaluation.

**Note**: Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.

## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🤔 Troubleshooting

| Problem                                                                                                                                      | Solution                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| (Windows) Invalid DOS bash script on WSL                                                                                                     | Run `dos2unix <script_name>` to convert script format          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |


## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
