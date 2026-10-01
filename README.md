# DevGPT: A morphodynamic language model for delineating cellular developmental behaviors from images([bioRxiv](https://www.biorxiv.org/content/10.1101/XXXX.XXXXXX))

Development is a fundamental process that a single cell divides into an organism with multiple fates, with complex spatial organization. Identifying and interpreting how individual cells change their properties is not only a central question in developmental biology but is also highly relevant to disease, in which pathological cells may acquire abnormal phenotypic features. Here, we introduce DevGPT, an artificial-intelligence-based foundation model that uses time-lapse images of cell morphology to predict cellular properties within developing embryos. Leveraging public datasets of Caenorhabditis elegans embryogenesis, we demonstrate that DevGPT can learn morphodynamic features associated with distinct tissues/organs and predict cellular developmental properties with high accuracy, exemplified by cell fate and division. Ablation experiments further reveal that the model integrates multiple biologically meaningful features, including cell shape, movement, and size. DevGPT provides a general framework for systematically learning from image-based cell morphology datasets, identifying discriminative phenotypic features and previously hidden cellular heterogeneity, and highlighting the power of artificial intelligence to infer cellular properties directly from dynamic morphological information and to uncover phenotypic signatures that may otherwise remain difficult to identify.

## Dataset Format

The input consists of ternary voxel maps organized per cell, with the expected structure below (numbers are examples only):

```
<raw_dir>/
└── <embryo>/                                  # one directory per embryo, e.g. WT_Sample1/
    └── <lineage>_<fate>/                      # one directory per cell (lineage name_terminal fate)
        ├── ABalaaaalal_Neuron_Cell_0010_WT_Sample1_193_segCell.npz
        ├── ABalaaaalal_Neuron_Cell_0010_WT_Sample1_194_segCell.npz
        ├── ABalaaaalal_Neuron_Cell_0010_WT_Sample1_195_segCell.npz
        └── ...                                # one cell forms a time series
```

- Each npz: a 3D `uint8` array (e.g. 128³) with values `{0,1,2}` — 0 background, 1 other embryonic cells, 2 target cell (key `arr` is read preferentially from the npz)

Python ≥ 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

### Step 1: NPME encoding (voxels → Cell Tokens)

The NPME autoencoder encodes the ternary voxel map of each time point into a continuous 32×4×4×4 morphological Token: the first 16 channels are anchored to interpretable geometric quantities (volume, surface area, shape indices, relative embryonic position, contact measures), while the last 16 channels encode higher-order morphological residuals, jointly carrying cell shape, relative position and overall embryonic posture. Each time point yields one `.npy` file (the original directory hierarchy is preserved):

```bash
python inference/infer_npme.py
```

Output structure:

```
<embedding_dir>/
└── <embryo>/
    └── <lineage>_<fate>/
        ├── <...>_193_segCell.npy           # (32, 4, 4, 4) float32
        └── ...
```

### Step 2: Building developmental sequences (Cell Tokens → morphological sentences)

Tokens of all frames of the same cell are stacked in temporal order into a `(T, 32, 4, 4, 4)` sequence (pure post-processing, no model involved):

```bash
python inference/embed2sequence.py
```

Output structure:

```
<output_dir>/
└── <embryo>/
    ├── <embryo>_<lineage>_<fate>.npy      # plain array (T, 32, 4, 4, 4) float32
    └── ...
```

### Step 3-1: Fate inference

```bash
python inference/inference4cellfate.py
```

Outputs `predictions.csv`, one row per cell sequence:

| Column | Meaning |
|---|---|
| file / true / pred | sequence file, ground-truth fate, predicted fate |
| is_known / confidence | whether the label is among the training classes, prediction confidence |
| top1–3_label / top1–3_prob | top-3 candidate fates and their probabilities |
| prob_\<class\> | probability of each fate class |

### Step 3-2: Time-to-division inference

The input sequence is truncated to a random-length prefix, and the model predicts the remaining number of frames until the next division:

```bash
python inference/inference4cellcycle.py
```

Outputs the Time-to-division prediction CSV, one row per cell sequence:

| Column | Meaning |
|---|---|
| file_name | sequence file name |
| gt_total_len / input_len | total number of frames / input prefix frames |
| gt_remain_len / pred_remain_len | ground-truth / predicted remaining frames |
| diff | predicted − ground truth |

## Training (optional)

```bash
torchrun --standalone --nproc_per_node=<N> pretrain/train_npme.py      # NPME autoencoder
python pretrain/train_backbone.py                                      # Transformer NTP pretraining
python finetune/finetune4cellfate.py                                  # fate-classification finetuning
python finetune/finetune4cellcycle.py                                 # time-to-division regression finetuning
```
