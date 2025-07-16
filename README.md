# MetricGold: Repurposing Diffusion-Based Image Generators for Monocular Metric Depth Estimation
[[Arxiv]](https://arxiv.org/abs/2411.10886), Paper will be updated soon with more information and tables.

We build upon the Marigold pipeline which predicts Monocular Relative Depth and train that pipeline for Log Scaled Monocular Metric Depth taking inspiration from DMD(Diffusion Models for Metric Depth). The idea is that we augment the metric depth in such a fashion that the VAE reconstruction is good and there's little need to finetune the VAE. Now we can leverage the LDM pipeline to finetune a model which predicts Metric Depth. All you need to do is a inverse of the log normalization function. We also aligned with the hypothesis of the Depth Anything V2, which states that Photorealistic Synthetic Data lacks sensor noise and biases and is better as Real data has sensor noise even when it has been preprocessed. Simulators can help us have a good initilisation and data redistillation like the SAM paper can improve the performance of our model due to domain expansion caused by distillation. Due to resource limitation we did not complete the second step of the Depth Anything v2 training reicipe where we pseudo label and retrain the model on a larger dataset.

function used for normalisation : D_aug = log(D / D_min) / log(D_max / D_min)

# TODO Update Repo Installation

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/AnshShah3009/MetricGold.git
cd MetricGold
```

### 💻 Dependencies

We provide several ways to install the dependencies.

1. **Using [Mamba](https://github.com/mamba-org/mamba)**, which can installed together with [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3). 

    Windows users: Install the Linux version into the WSL.

    After the installation, Miniforge needs to be activated first: `source /home/$USER/miniforge3/bin/activate`.

    Create the environment and install dependencies into it:

    ```bash
    mamba env create -n metricgold --file environment.yaml
    conda activate metricgold
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

<!-- By default, the [checkpoint](https://huggingface.co/prs-eth/marigold-v1-0) is stored in the Hugging Face cache. -->
The `HF_HOME` environment variable defines its location and can be overridden, e.g.:

```bash
export HF_HOME=$(pwd)/cache
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

Run training script

```bash
python train.py --config config/train_metricgold.yaml
```

Resume from a checkpoint, e.g.

```bash
python train.py --resume_run output/metricgold_base/checkpoint/latest
```

## 📚 Citation

This project uses the following work:

Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R. C., & Schindler, K. (2024). *Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024

BibTeX:
```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Ke, Bingxin and Obukhov, Anton and Huang, Shengyu and Metzger, Nando and Daudt, Rodrigo Caye and Schindler, Konrad},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  month={June},
  pages={XXXX-XXXX}
}
