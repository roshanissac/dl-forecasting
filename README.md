# Empirical Analysis of NBeats,N-Hits and TFT Architectures

This is an empirical analysis NBeats,N-Hits and TFT Architectures on Rossman Sales Dataset as part of DS8013 Deep Learning Course Project for Toronto Metropolitan University's(TMU) Data Science Master's (MSc) program.

The primary purpose of this project is to do an empirical analysis of rossman dataset leveraging novel forecasting frameworks such as N-BEATS, N-HITS, and Temporal Fusion Transformers(TFT).The project aims to study the performance of these frameworks by comparing it with a suitable baseline with a focus on sales prediction at Rossman Stores.

# Data

The original raw source of the data used for this experiment taken from Kaggle from the below link,

* [Train Dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data)

These datasets are downloaded and stored under **datasets/raw/** folder.

# Setup

This project requires **high memory**.The results of this project is obtained by running in Google Colab environment with High Memory.The PyTorch Forecasting library triggered some bugs when ran in GPU ,Hence the models was trained on CPU only.

Below are the steps to run the experiments,

1. Clone this repository to your local machine.
2. Create a conda or virtual environment by executing below command on anaconda terminal(In Windows) or in Terminal(Mac).Make sure you have installed anaconda.
   ```
   conda create -n dl-project python=3.10
   ```
3. Activate the created environment by executing below command,
   ```
   conda activate dl-project
   ```
4. Go to the root folder of the project and Install the packages/requirements by executing the below command.
   ```
   pip  install -r requirements.txt
   ```
5. **main.py** is the main file and too run the experiments please follow below commands,
   
   To run experiment with TFT model,execute the below commands,
   ```
   python main.py --model "tft" --no_of_epochs 50  
   ```
   To run experiment with N-BEATS model,execute the below command,
   ```
   python main.py --model "nbeats" --no_of_epochs 50  
   ```
   To run experiment with N-HITS model,execute the below command,
   ```
   python main.py --model "nhits" --no_of_epochs 50  
   ```
   To run all experiments together you can add *--all* flag to the command as below,In this case for --model parameter you can pass empty string.
   ```
   python main.py --model " " --no_of_epochs 50  --all
   ```

# Notes
1. The first step of the experiment will be EDA and Preprocessing irrespective which model you selected to run.
2. Preprocessed data is stored under the folder **datasets/preprocessed/**
3. All the charts generated during EDA is stored under the folder **charts/EDA/**
4. The core files are located under the folder **helpers/**
5. Forecasting charts generated by each model is stored under **charts/nbeats/**,**charts/nhits/**,**charts/tft/** folders.

# References

1. PyTorch Forecasting:[link](https://pytorch-forecasting.readthedocs.io/en/stable/index.html)
2. Rossman Sales Dataset:[link](https://www.kaggle.com/competitions/rossmann-store-sales/data)
3. Oreshkin, B. N., Carpov, D., Chapados, N., \& Bengio, Y.  } (2020) N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. Proceedings of the International Conference on Learning Representations (ICLR)}: [link](https://arxiv.org/abs/1905.10437)
4. Lim, B., Arık, S. Ö., Loeff, N., \& Pfister, T.  } Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:[link](https://arxiv.org/abs/1912.09363)
5. Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski } (2021). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting:[link](https://arxiv.org/abs/2201.12886)

