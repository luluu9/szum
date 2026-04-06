# Audio Classification Project

Welcome to the Audio Classification Project! This repository contains the complete pipeline for building our machine learning model. It covers every step of the process, starting from raw data preparation and statistical analysis, through data processing, all the way to the training and evaluation of the final classification model.

Our goal is to build a robust model capable of classifying audio frames into three main categories: `Music`, `Speech`, and `Other` (background noise).

## Prerequisites & Data Sources

To execute the scripts in this repository, you must first download the required audio datasets and place them in the correct structure within the `datasets/` directory. The pipeline is designed around the following three corpora:

1. **GTZAN Music Genre Classification Dataset**
   - Source: [Kaggle - GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
   - *Used for balancing the Music class.*

2. **MUSAN (A Music, Speech, and Noise Corpus)**
   - Source: [OpenSLR - MUSAN](https://www.openslr.org/17/)
   - *Used for providing high-quality English Speech, supplementary Music, and background Noise files.*

3. **DEMAND (Diverse Environments Multichannel Acoustic Noise Database)**
   - Source: [Kaggle - DEMAND](https://www.kaggle.com/datasets/aanhari/demand-dataset/data)
   - *Used for representing diverse environmental audio for the Other class.*

Please ensure the data is properly downloaded and extracted. Then, you can proceed to the `data_preparation` directory to execute the dataset merge and balancing scripts.
