#!/bin/bash

# Define the URL
URL_HATE_ZIP="https://zenodo.org/records/7799469/files/hate_videos.zip"
URL_NON_HATE_ZIP="https://zenodo.org/records/7799469/files/non_hate_videos.zip"
URL_ANNOTATION="https://zenodo.org/records/7799469/files/HateMM_annotation.csv"
URL_README="https://zenodo.org/records/7799469/files/readme.txt"

RAW_TEXT_HATE="./data/raw/text/hateful"
RAW_TEXT_NON_HATE="./data/raw/text/non-hateful"
RAW_VID_HATE="./data/raw/videos/hateful"
RAW_VID_NON_HATE="./data/raw/videos/non-hateful"
DOCS="./data/raw"
CLEAN_DIR="./data/clean"

# Create the directory structure
mkdir -p "$DOCS" "$RAW_TEXT_HATE" "$RAW_TEXT_NON_HATE" "$RAW_VID_HATE" "$RAW_VID_NON_HATE" "$CLEAN_DIR"

# download files
curl -L "$URL_HATE_ZIP" -o "$RAW_VID_HATE/hate_videos.zip"
curl -L "$URL_NON_HATE_ZIP" -o "$RAW_VID_NON_HATE/non_hate_videos.zip"
curl -L "$URL_ANNOTATION" -o "$DOCS/HateMM_annotation.csv"   # Move later if needed
curl -L "$URL_README" -o "$DOCS/readme.txt"

# unzip the videos
unzip -o "$RAW_VID_HATE/hate_videos.zip" -d "$RAW_VID_HATE"
unzip -o "$RAW_VID_NON_HATE/non_hate_videos.zip" -d "$RAW_VID_NON_HATE"

# remove zip files after extraction to save space
rm "$RAW_VID_HATE/hate_videos.zip"
rm "$RAW_VID_NON_HATE/non_hate_videos.zip"