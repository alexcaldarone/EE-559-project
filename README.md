# EE-559-project
Final mini-project for EE-559 course at EPFL (Spring 2025)


### Repo structure
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

### How to reproduce