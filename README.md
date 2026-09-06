# 🏁 Grid2Flag - F1 Top-10 Predictor

A manually implemented neural network-based Formula 1 race outcome predictor that forecasts whether a driver will finish in the top 10 based on their performance in the first 10 laps of a race.

**Key Achievement:** 81.7% accuracy with 87.9% AUC using circuit-aware Z-score features.

---

## 🎯 Features

✅ **Circuit-Aware Predictions** - Z-score normalization relative to circuit averages  
✅ **Custom Neural Network** - Built from scratch with NumPy (no TensorFlow/PyTorch)  
✅ **Interactive Streamlit App** - Easy-to-use interface with presets and manual input  
✅ **Easy Mode Presets** - Podium, Above Average, Average, Poor, and Random scenarios  
✅ **Circuit-Specific Context** - Driver and team performance at selected circuit  
✅ **Real-Time Predictions** - Instant top-10 probability with confidence levels  

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.70% |
| **Precision** | 84.95% |
| **Recall** | 86.54% |
| **F1-Score** | 85.74% |
| **AUC** | 87.90% ⭐ |
| **Specificity** | 73.26% |

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/keyyril/Grid2Flag.git
cd Grid2Flag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Download the [Kaggle Formula 1 Dataset](https://www.kaggle.com/datasets/melissamonfared/formula-1) and place in `data/raw/`

### 4. Run Notebooks (One-Time Setup)
```bash
# Feature Engineering + Z-score calculation
jupyter notebook notebooks/02_feature_engineering.ipynb

# Data Preparation
jupyter notebook notebooks/03_prepare_for_nn.ipynb

# Model Training
jupyter notebook notebooks/04_train_model.ipynb

# Statistics Calculation
jupyter notebook notebooks/06_calculate_stats.ipynb
```

### 5. Launch App
```bash
streamlit run app.py
```

Opens at: `http://localhost:8501`

---

## 📁 Project Structure

```
Grid2Flag/
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── models/                         # Neural network implementation
│   ├── __init__.py
│   ├── layers.py                   # Dense layer
│   ├── activations.py              # ReLU, Sigmoid
│   ├── loss.py                     # Binary Cross-Entropy
│   └── neural_network.py           # Main NN class
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_explore_data.ipynb       # EDA
│   ├── 02_feature_engineering.ipynb # Z-score features
│   ├── 03_prepare_for_nn.ipynb     # Data normalization
│   ├── 04_train_model.ipynb        # Model training
│   ├── 05_model_evaluation.ipynb   # Performance metrics
│   └── 06_calculate_stats.ipynb    # Circuit/driver/team stats
│
└── data/
    ├── raw/                        # Original CSVs 
    └── processed/                  # Generated files
        ├── trained_model.pkl
        ├── normalization.pkl
        ├── circuit_baselines.json
        ├── circuit_stats.json
        ├── driver_stats_by_circuit.json
        └── team_stats_by_circuit.json
```

---

## 🎮 How to Use the App

### 1. **Choose Input Method**
- **Easy Mode:** Click a preset (Podium, Above Avg, Average, Poor, Random)
- **Manual:** Enter exact lap times and data
- **Mix:** Use preset then modify values

### 2. **Review Context**
- Circuit average lap time
- Driver performance at circuit
- Team performance at circuit

### 3. **Make Prediction**
Click "🔮 Make Prediction" button

### 4. **See Results**
- TOP-10 or OUTSIDE TOP-10
- Probability percentage
- Confidence level (High/Medium/Low)
- Feature breakdown

---

## 🧠 Model Architecture

```
Input: 7 features (normalized)
    ↓
Dense(16) + ReLU
    ↓
Dense(8) + ReLU
    ↓
Dense(1) + Sigmoid
    ↓
Output: Probability (0-1)
```

### Features

1. **laps_completed_early** - Laps completed in first 10
2. **avg_lap_time_relative_zscore** ⭐ - Z-score relative to circuit (NEW!)
3. **lap_time_std** - Standard deviation of lap times
4. **lap_time_min** - Fastest lap achieved
5. **pace_degradation** - Slowing down over laps
6. **avg_position_early** - Average position in first 10
7. **circuit_normalized** - Circuit ID (normalized)

### The Z-Score Magic

```
Z = (driver_lap_time - circuit_average) / circuit_std

Example:
- 95s at Monaco (avg 94s): z = +0.67 (normal) ✓
- 95s at Monza (avg 82s): z = +6.5 (very slow) ✗
```

This makes the model **circuit-aware** and improves accuracy by 5-8%!

---

## 📊 Data Sources

- **Dataset:** [Kaggle - Melissa Monfared Formula 1](https://www.kaggle.com/datasets/melissamonfared/formula-1)
- **Coverage:** 1,125 races, 859 drivers, 77 circuits
- **Training Samples:** 8,468

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | NumPy (from scratch!) |
| **App Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Language** | Python 3.8+ |

---
## 🤔 FAQ

**Q: Do I need to download the Kaggle dataset?**  
A: Yes, download from [Kaggle F1 Dataset](https://www.kaggle.com/datasets/melissamonfared/formula-1) and extract to `data/raw/`

**Q: Can I run just the app without the notebooks?**  
A: No, you need to run notebooks first to generate the processed data files.

**Q: How long does training take?**  
A: ~30-60 seconds on CPU

**Q: Can I improve the model?**  
A: Yes! Try: more laps (20-30), weather data, pit stop info, driver form tracking

---
**Built with ❤️ and NumPy** 🏁
