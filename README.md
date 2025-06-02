# EE-559-project
_Final mini-project for EE-559 course at EPFL (Spring 2025)_

**Project description:** In this work, we present a multimodal approach to detecting hateful content in short videos by leveraging CLIP embeddings and finetuning them. We demonstrate that swapping or masking one modality (visual or textual) can significantly degrade classification performance -- an effect we term ``hateful content masking" -- and show that incorporating such adversarially masked samples into training enhances model robustness. Our experiments reveal that textual embeddings are particularly informative for hate detection.


## Repo structure
```
project-root/
├── configs/                         # Config files
│
├── data/                            # All datasets (not committed to Git)
│   ├── clean/                       # Preprocessed data
│   │   ├── audios/                 # Extracted audio files
│   │   ├── frames/                 # Extracted video frames
│   │   ├── texts/                  # Transcriptions (from Whisper)
│   │   └── videos/                 # Snippets after segmentation
│   └── raw/                         # Raw original data (source files)
│       ├── text/
│       ├── videos/
│       └── HateMM_annotation.csv   # Annotations file
│   ├── embeddings/                 # embeddings from clip
│   ├── finetuned_embeddings/       # embeddings from finetuned clip
│   └── embeddings_transformed/     # embeddings of augmented data (not used)
│
├── logs/                            # Logging output
│
├── models/                          # Saved models and checkpoints
│   └── checkpoints/
│       ├── clip_finetune/           # finetuned clip weights
│       ├── robust_training/         # weights of robust models
│       ├── train_normal_test_on_shuffled/
│       └── training_simple/ 
│
├── notebooks/ 
│
├── results/                         # Outputs, metrics
│
├── scripts/                         # CLI scripts 
│
├── src/                             # Source code for the project
│   ├── data_processing/    
│   ├── model/         
│   └── utils/                      
│
├── .gitignore                       # Git ignore rules
├── data_download.sh                # Shell script to download data
├── Dockerfile                       # For containerized setup (optional)
├── requirements.txt                # Python dependencies
└── README.md                        # Project documentation

```

## Setting up the environment

In order to reproduce the experiments the Docker image needs to be built. You can do this by building it using the Dockerfile provided. 

If running on the runai EPFL cluster, 

- To build an interactive job use the following image:

    ```bash
    runai submit --image registry.rcp.epfl.ch/ee-559-acaldaro/my-toolbox:v0.2 [rest of command]
    ```

- To use a regular job:

    ```bash
    runai submit --image registry.rcp.epfl.ch/ee-559-acaldaro/my-toolbox:v0.2 [rest of command for rcp] \
        --command pip install -r requirements.txt && \
        [python command]
    ```
    for the possible python commands needed to reproduce the results of the project, see below.

## How to reproduce

1. **Downloading Data**
    
    First, download the data from the source using the provided script. This will download the original data used for the project and create the required directory structure to store the raw and processed data.
    ```bash
    ./data_download.sh
    ```

2. **Preprocessing Data**

    Next, run the following command to preprocess the data. This will extract audio, video, text and image frames from the raw data and save them in the `data/clean` directory.
    ```bash
    python3 -m scripts.data_preprocessing --steps all --model base --seed 42
    ```

    Arguments:
    
    - `--steps`: One or more pipeline stages to run. Options:
        - `snippets` – extract video snippets
        - `audio` – extract audio from video snippets
        - `text` – transcribe audio with Whisper
        - `frames` – extract frames from video snippets
        - `all` – run the entire pipeline (default)
    - `--model`: (Optional) Whisper model to use. Choose from: `tiny`, `base`, `small`, `medium`, `large`
    - `--seed`: (Optional) Random seed for reproducibility (default: `42`)
    - `--overwrite`: (Optional) If passed, will overwrite existing files instead of skipping them

    Make sure to run this command from the **project root**, not from inside the `scripts/` folder.

3. **Obtaining embeddings with CLIP**

    Once the clean data has been obtained, you can extract the embeddings using the CLIP model by running:
    ```bash
    python3 -m scripts.compute_embeddings
    ```

4. **Finetuning CLIP**

    In order to finetune CLIP to obtain better embeddings, run the following command:
    ```bash
    python3 -m scripts.finetune_clip --config configs/finetune_clip.yml
    ```
    where `configs/finetune_clip.yml` is the config file containing the setting for the finetuning of the model.

5. **Training + Model Evaluation**

    To train the model and evaluate its performance run:
    ```bash
    python3 -m scripts.training --config configs/training_simple.yml \
        --model_name [model_name] \
        --finetune \
        --test_on_shuffled \
        --modality_to_shuffle_in_test [modality]
    ```
    where:

    - `configs/training_simple.yml` contains the configuration file with the traning hyperparameters
    
    - `--model_name [model_name]` indicates the names of the model to train. Currently options are: `BinaryClassifer`, `CrossModalFusion` and `?`
    
    - `--finetune` is an optional boolean flag used to indicate whether to train using the embeddings from the finetuned CLIP
    
    - `--test_on_shuffled` is a boolean flag that indicates whether to use the data with swapped modalities for hateful content on the test set
    
    - `--modality_to_shuffle_in_test [modality]` indicates what modality to shuffle on the test set. Can be `text` or `image`.

6. **"Robust" training**

    In order to include the data with the swapped modalities into the traning set and make the model robust to this type of data corruption, run the following command:
    ```bash
    python3 -m scripts.training_robust --config configs/training_robust.yml \
        --finetuned \
        --modality_to_shuffle [modality] \
        --test_on_shuffled \
        --modality_to_shuffle_in_test [modality] 
    ```
    where:

    - `configs/training_robust.yml` contains the configuration file with the robust traning hyperparameters
    
    - `--finetune` is an optional boolean flag used to indicate whether to train using the embeddings from the finetuned CLIP

    - `--modality_to_shuffle [modality]` indicates what modality to shuffle on the train set for the further robust training epochs. Can be `text` or `image`.
    
    - `--test_on_shuffled` is a boolean flag that indicates whether to use the data with swapped modalities for hateful content on the test set
    
    - `--modality_to_shuffle_in_test [modality]` indicates what modality to shuffle on the test set. Can be `text` or `image`.

## Results

The results of the various training runs are saved in the `results/` directory. See the [relevant README](./results/README.md) file for the result directory structure.

The `notebooks/` directory contains the notebook used to analyse the results and produce the plots presented on the poster.


## Contributors

- Alex John Caldarone, `alex.caldarone@epfl.ch`
- Stephan Hengl, `stephan.hengl@epfl.ch`
- Samuel Ahou, `samuel.ahou@epfl.ch`