# EE-559-project
Final mini-project for EE-559 course at EPFL (Spring 2025)


## Repo structure
```
project-root/
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
│
├── logs/                            # Logging output
│
├── models/                          # Saved models and checkpoints
│   └── checkpoints/
│
├── results/                         # Outputs, metrics
│
├── scripts/                         # CLI scripts
│
├── src/                             # Source code for the project
│   ├── data_processing/            
│   └── utils/                      
│
├── .gitignore                       # Git ignore rules
├── data_download.sh                # Shell script to download data
├── Dockerfile                       # For containerized setup (optional)
├── requirements.txt                # Python dependencies
└── README.md                        # Project documentation

```

## How to reproduce

1. **Downloading Data**
    
    First, download the data from the source using the provided script. This will download the original data used for the project and create the required directory structure to store the raw and processed data.
    ```bash
    ./data_download.sh
    ```

2. **Preprocessing Data**

    Next, run the following command to preprocess the data. This will extract audio, video, text and image frames from the raw data and save them in the `data/clean` directory.
    ```bash
    python3 -m src.data_preprocessing
    ```