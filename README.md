# anomally_detection_nlp
#introduction
Spacecraft and satellite systems generate large volumes of telemetry data from multiple sensors during operation. This project aims to develop an intelligent anomaly detection system that analyzes multivariate spacecraft telemetry data using deep learning techniques.
#Architecture

Input (4 channels) 
    ↓
LSTM Encoder (80 hidden units, 2 layers)
    ↓
Latent Bottleneck (40 dims)
    ↓
LSTM Decoder (80 hidden units, 2 layers)
    ↓
Output (4 channels reconstructed)

#dataset
Source: NASA/University of Cincinnati
Test: 2nd_test (984 files)
Channels: 4 bearing vibration sensors
Size: ~20M data points

#installation
telemanom_lstm/
├── data/
│   ├── 2nd_test/
│   │   ├── 2004.02.12.10.32.39  (no .txt extension)
│   │   ├── 2004.02.12.10.42.39
│   │   └── ... (984 files)
│   └── 3rd_test/
│       ├── 2004.03.04.09.27.46
│       └── ... (4448 files)
├── config.py
├── model.py
└── ...

#Project structure
telemanom_lstm/
├── data/                # Place bearing data here (see Installation)
│   ├── 2nd_test/
│   │   └── txt/        # 984 bearing data files
│   └── 3rd_test/
├── config.py            # Hyperparameters and paths
├── model.py             # Telemanom LSTM architecture
├── data_loader.py       # Data preprocessing
├── train.py             # Training loop
├── evaluate.py          # Evaluation and metrics
├── run_all.py           # Master workflow script
├── demo_corrupt.py      # Runs the model on artificial data
├── demo_model.py        # Runs the model on test data
├── analyze_splits.py    # Get the split marks for training and test data
├── requirements.txt     # Dependencies
├── .gitignore           # Git ignore rules
├── checkpoints/         # Saved models (auto-created)
└── results/             # Plots and results (auto-created)
