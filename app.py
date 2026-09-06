"""
Grid2Flag: Streamlit app for F1 race outcome prediction with circuit and statistics.

This app predicts whether an F1 driver will finish in the top 10 based on
their performance in the first 10 laps of a race.

Yea sure- theres better f1 preds, like podium or race winner, but main thing is i
want to manually implement the neural network by hand and not just import-to get
a better understanding (also cuz its cool :)

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

#======================================================================================
# Page Configuration
#======================================================================================

st.set_page_config(
    page_title = "Grid2Flag - F1 Top 10 Predictor",
    page_icon = "🏁",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

#======================================================================================
# Custom Styling
#======================================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #FF1801;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF1801;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html = True)

#======================================================================================
# Cache & load data
#======================================================================================

@st.cache_resource

def load_model_and_data():
    """Load all required data & model"""
    try:
        # Add parent directory for path imports
        import sys
        sys.path.insert(0, '.')
        from models import NeuralNetwork

        # Load trained model
        with open("data/processed/trained_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load normalization parameters
        with open("data/processed/normalization.pkl", "rb") as f:
            norm_params = pickle.load(f)

        # Load circuit baseline (z-score calculation)
        with open("data/processed/circuit_baselines.json", "rb") as f:
            circuit_baselines = json.load(f)

        # Load statistics for context
        with open("data/processed/circuit_stats.json", "rb") as f:
            circuit_stats = json.load(f)

        # Load driver stats
        with open("data/processed/driver_stats_by_circuit.json", "r") as f:
            driver_stats = json.load(f)

        # Load team stats
        with open("data/processed/team_stats_by_circuit.json", "r") as f:
            team_stats = json.load(f)

        return model, norm_params, circuit_baselines, circuit_stats, driver_stats, team_stats

    except FileNotFoundError as e:
        st.error(f"❌ Error loading file: {e}")
        st.info("Make sure notebook 02 till 06 gang <3")
        st.stop()

# Load all data
model, norm_params, circuit_baselines, circuit_stats, driver_stats, team_stats = load_model_and_data()

# Initialize session state
if "preset" not in st.session_state:
    st.session_state.preset = None
if "random_values" not in st.session_state:
    st.session_state.random_values = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_z_score_for_circuit(lap_time_ms, circuit_name):
    """
        Calculate Z-score relative to circuit average
        
        Formula: z = (driver_time - circuit_mean) / circuit_std
    """
    if circuit_name not in circuit_baselines:
        return 0.0

    baseline = circuit_baselines[circuit_name]
    circuit_mean = baseline["mean"]
    circuit_std = baseline["std"]

    if circuit_std == 0:
        return 0.0

    z_score = (lap_time_ms - circuit_mean) / circuit_std
    return z_score

def normalize_features(features, norm_params):
    """Apply normalization using training sets statistics """
    mean = norm_params['mean']
    std = norm_params['std']
    return (features - mean) / std

def make_predictions(features_normalized, threshold = 0.5):
    """Make preds on normalized features"""
    probability = model.predict(features_normalized.reshape(1, -1))[0, 0]
    prediction = 1 if probability >= threshold else 0
    return probability, prediction

def get_circuit_context(circuit_name):
    """Get circuit context"""
    if circuit_name in circuit_stats:
        return circuit_stats[circuit_name]
    return None

def get_driver_context(driver_name, circuit_name):
    """Get driver context for specific circuit"""
    if driver_name in driver_stats:
        if circuit_name in driver_stats[driver_name]:
            return driver_stats[driver_name][circuit_name]
    return None

def get_team_context(team_name, circuit_name):
    """Get team context for specific circuit"""
    if team_name in team_stats:
        if circuit_name in team_stats[team_name]:
            return team_stats[team_name][circuit_name]
    return None

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

# with st.sidebar:
#    st.markdown("#🛠 Configuration")

#   threshold = st.slider(
#        "Decision Threshold",
#        min_value = 0.0,
#        max_value = 1.0,
#        value = 0.5,
#        step = 0.01,
#        help = "Adjust the probability threshold for top 10 prediction"
#    )
#
#    if abs(threshold - 0.531) < 0.5:
#       st.success("✅ Near optimal threshold (0.531)")

# ============================================================================
# MAIN APP
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class = "main-header">▀▄▀▄▀▄ Grid2Flag ▀▄▀▄▀▄</p>', unsafe_allow_html = True)
    st.markdown('<p class="sub-header">F1 Top-10 Predictor (First 10 Laps)</p>', unsafe_allow_html = True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🔮Predictions", "🔎Model Info", "👀How It Works", "💡About"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================

with tab1:
    st.markdown("## 🔮Make a prediction🧙‍♂️")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💻Race Setup")

        circuit_name = st.selectbox(
            "Circuit",
            options = sorted (circuit_baselines.keys())
        )

        team_name = st.selectbox(
            "Team",
            options = sorted (team_stats.keys())
        )

        driver_options = ["-- Select Driver --"] + sorted(driver_stats.keys())
        driver_name = st.selectbox("Driver", options=driver_options)
        if driver_name == "-- Select Driver --":
            driver_name = None

    with col2:
        st.markdown("##  🔁First 10 Laps Data🔁")

        # EASY MODE + RANDOM BUTTON (NO CUSTOM BUTTON)
        st.markdown("**⚡ Quick Start (Optional):**")
        col_a, col_b, col_c, col_d, col_e = st.columns([1, 1, 1, 1, 1])
        
        with col_a:
            if st.button("🥇Podium", use_container_width = True):
                st.session_state.preset = "podium"
                # Clear random values when switching presets
                st.session_state.random_values = None
        with col_b:
            if st.button("👍 Superior", use_container_width = True):
                st.session_state.preset = "above_avg"
                st.session_state.random_values = None
        with col_c:
            if st.button("👌 Moderate", use_container_width = True):
                st.session_state.preset = "average"
                st.session_state.random_values = None
        with col_d:
            if st.button("👎 Struggling", use_container_width = True):
                st.session_state.preset = "poor"
                st.session_state.random_values = None
        with col_e:
            if st.button("🎲Random", use_container_width = True):
                st.session_state.preset = "random"
                # Generate random values ONCE and store them
                circuit_avg_ms = circuit_baselines[circuit_name]["mean"]
                circuit_avg_sec = circuit_avg_ms / 1000
                st.session_state.random_values = {
                    'lap_sec': round(np.random.uniform(circuit_avg_sec - 2.0, circuit_avg_sec + 3.0), 1),
                    'position': round(np.random.uniform(1.0, 15.0), 1),
                    'degradation': round(np.random.uniform(-2000.0, 2000.0), 0),
                    'std': round(np.random.uniform(500.0, 4000.0), 0),
                    'fastest': round(np.random.uniform(circuit_avg_ms - 4000, circuit_avg_ms + 2000), 0)
                    }
        
        st.caption("💡 Click a preset to auto-fill, or skip and enter manually below")
        
        # Get circuit average for context-aware presets
        circuit_avg_ms = circuit_baselines[circuit_name]["mean"]
        circuit_avg_sec = circuit_avg_ms / 1000
        
        # Set defaults based on preset
        if "preset" in st.session_state and st.session_state.preset:
            preset = st.session_state.preset

            if preset == "podium":
                lap_default_sec = round(circuit_avg_sec - 1.5, 1)
                position_default = 2.0
                degradation_default = -1000.0
                std_default = 500.0
                fastest_default = round((circuit_avg_ms - 3000), 0)

            elif preset == "above_avg":
                lap_default_sec = round(circuit_avg_sec - 0.8, 1)
                position_default = 4.0
                degradation_default = -500.0
                std_default = 1000.0
                fastest_default = round((circuit_avg_ms - 1500), 0)

            elif preset == "average":
                lap_default_sec = round(circuit_avg_sec, 1)
                position_default = 7.0
                degradation_default = 0.0
                std_default = 1500.0
                fastest_default = round((circuit_avg_ms - 500), 0)

            elif preset == "poor":
                lap_default_sec = round(circuit_avg_sec + 2.0, 1)
                position_default = 12.0
                degradation_default = 1000.0
                std_default = 3000.0
                fastest_default = round((circuit_avg_ms + 1000), 0)
            
            elif preset == "random":
                # Use stored random values (generated ONCE and never regenerated)
                if st.session_state.random_values is None:
                    st.session_state.random_values = {
                        'lap_sec': round(np.random.uniform(circuit_avg_sec - 2.0, circuit_avg_sec + 3.0), 1),
                        'position': round(np.random.uniform(1.0, 15.0), 1),
                        'degradation': round(np.random.uniform(-2000.0, 2000.0), 0),
                        'std': round(np.random.uniform(500.0, 4000.0), 0),
                        'fastest': round(np.random.uniform(circuit_avg_ms - 4000, circuit_avg_ms + 2000), 0)
                    }
                
                lap_default_sec = st.session_state.random_values['lap_sec']
                position_default = st.session_state.random_values['position']
                degradation_default = st.session_state.random_values['degradation']
                std_default = st.session_state.random_values['std']
                fastest_default = st.session_state.random_values['fastest']

        else:
            lap_default_sec = round(circuit_avg_sec, 1)
            position_default = 7.0
            degradation_default = -500.0
            std_default = 2000.0
            fastest_default = round((circuit_avg_ms - 500), 0)
        
        st.markdown("**Manual Input:**")
        
        # Input fields - users can modify ANY of these!

        laps_completed = st.slider("Laps Completed (0-10)", 0, 10, 10)

        lap_time_sec = st.number_input("Average Lap Time (seconds)", 60.0, 300.0, lap_default_sec, 0.1)

        lap_time_ms = lap_time_sec * 1000

        lap_time_std = st.number_input("Lap Time Std Dev (ms)", 0.0, 10000.0, std_default, 100.0)

        lap_time_min = st.number_input("Fastest Lap (ms)", 0.0, 150000.0, fastest_default, 100.0)

        pace_degradation = st.number_input("Pace Degradation (ms)", -50000.0, 50000.0, degradation_default, 100.0)
        
        avg_position = st.number_input("Average Position (1-20)", 1.0, 20.0, position_default, 0.1)

    st.markdown("---")

    # Circuit Context (SIMPLIFIED - no top10_rate or podium_rate)
    st.markdown("### 🏁 Circuit Context🗺️")
    
    # Use circuit_stats for display
    if circuit_name in circuit_stats:
        circuit_info = circuit_stats[circuit_name]
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("⏱️ Circuit Avg Lap", f"{circuit_info['avg_lap_time_sec']}s")
        
        with col2:
            st.metric("🏁 Races Here", f"{circuit_info['races_count']}")
        
        # Track comparison using circuit_baselines for Z-score
        st.markdown("### ✍🏻 Your Input vs Circuit Average⏳")
        
        baseline = circuit_baselines[circuit_name]
        circuit_avg_ms = baseline["mean"]
        circuit_std_ms = baseline["std"]
        
        z_score = calculate_z_score_for_circuit(lap_time_ms, circuit_name)
        diff_ms = lap_time_ms - circuit_avg_ms
        diff_sec = diff_ms / 1000
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if diff_ms < 0:
                st.success(f"⚡ **{abs(diff_sec):.2f}s FASTER** than avg⚡")
            elif diff_ms > 0:
                st.warning(f"🐢 **{diff_sec:.2f}s SLOWER** than avg🐌 (pick up your pace)")
            else:
                st.info("😐 **AT AVERAGE**😑")
        
        with col2:
            st.metric("Z-Score", f"{z_score:.2f}")
        
        with col3:
            if abs(z_score) < 0.5:
                st.success("⚔️ Competitive")
            elif abs(z_score) < 1.5:
                st.info("🤝 Reasonable")
            else:
                st.warning("🚨 Off average")
            
    st.markdown("---")

    # Driver Context (Circuit-Specific)
    if driver_name:
        driver_info = get_driver_context(driver_name, circuit_name)
        
        if driver_info:
            st.markdown(f"### 🙎🏻‍♂️ {driver_name} Performance at {circuit_name}")
            col1, col2, col3 = st.columns(3)
        
            with col1:
                avg_finish = driver_info.get("avg_finish")
                st.metric(
                "Average Finish",
                f"{avg_finish:.1f}" if avg_finish is not None else "N/A"
                )
        
            with col2:
                top10_rate = driver_info.get("top10_rate")
                st.metric(
                "Top-10 Rate",
                f"{top10_rate:.1f}%" if top10_rate is not None else "N/A"
                )
        
            with col3:
                wins = driver_info.get("wins")
                st.metric(
                "Wins",
                f"{int(wins)}" if wins is not None else "N/A"
                )
        
    st.markdown("---")
           
    # Team Context
    if team_name:
        team_info = get_team_context(team_name, circuit_name)

    if team_info:
        st.markdown(f"### 🏆 {team_name} Performance at {circuit_name}")
        col1, col2, col3 = st.columns(3)

    with col1:
        avg_finish = team_info.get("avg_finish")
        st.metric(
            "Average Finish",
            f"{avg_finish:.1f}" if avg_finish is not None else "N/A"
        )

    with col2:
        top10_rate = team_info.get("top10_rate")
        st.metric(
            "Top-10 Rate",
            f"{top10_rate:.1f}%" if top10_rate is not None else "N/A"
        )

    with col3:
        wins = team_info.get("wins")
        st.metric(
            "Team Wins",
            f"{int(wins)}" if wins is not None else "N/A"
        )

    st.markdown("---")

    # Prediction
    if st.button("🔮Make Prediction🧙‍♂️", use_container_width = True, type = "primary"):

        # Create feature vector
        features_raw = np.array([[
            laps_completed,
            calculate_z_score_for_circuit(lap_time_ms, circuit_name),
            lap_time_std,
            lap_time_min,
            pace_degradation,
            avg_position,
            (list(circuit_baselines.keys()).index(circuit_name) - len(circuit_baselines.keys()) / 2) /
            (len(circuit_baselines.keys()) / 2)
        ]], dtype = np.float32)

        # Normalize feature
        features_normalized = normalize_features(features_raw, norm_params)

        # Make predictions
        probability, prediction = make_predictions(features_normalized, threshold = threshold)

        # Display result
        st.markdown("---")
        st.markdown("## 🔮Prediction Result🧙‍♂️")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if prediction == 1:
                st.success("### 🔟Top-10 Finish")
                st.markdown(f"**Probability: {probability * 100:.1f}%**")

            else:
                st.error("### 🥴Outside Top-10 Finish")
                st.markdown(f"**Probability of top-10: {probability * 100:.1f}%**")

        # Confidence
        st.markdown("### 📊 Confidence")

        if probability > 0.8 or probability < 0.2:
            st.success("🟢 High Confidence")

        elif probability > 0.6 or probability < 0.4:
            st.warning("🟡 Medium Confidence")

        else:
            st.info("🟠 Low Confidence (Close Call)")
                
        st.progress(min(probability, 1.0))
                
        # Feature breakdown
        st.markdown("### 🔍 Feature Breakdown")
        feature_names = norm_params['feature_names']
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': features_normalized[0]
        })

        st.dataframe(feature_df, use_container_width = True) # try use width = "stretch" if problem

# ============================================================================
# TAB 2: MODEL INFO
# ============================================================================

with tab2:
    st.markdown("## Model Architecture")

    col1, col2 = st.columns(2)

    with col1:
            st.markdown("### 🧠 Network Architecture")
            st.markdown("""
            ```
            Input:  7 features (normalized)
                ↓
            Dense(16) + ReLU
                ↓
            Dense(8) + ReLU
                ↓
            Dense(1) + Sigmoid
                ↓
            Output: Probability (0-1)
            ```
            """)

    with col2:
        st.markdown("### 📈 Training Configuration")
        st.markdown("""
        - Loss: Binary Cross-Entropy
        - Optimizer: Mini-batch SGD
        - Learning Rate: 0.01
        - Batch Size: 32
        - Epochs: 100
            
        **Performance:**
        - Accuracy: 81.70%
        - Precision: 84.95%
        - Recall: 86.54%
        - F1: 85.74%
        - AUC: 87.90%
        - Specificity: 73.26%
        """)
        
        st.markdown("---")
        
        st.markdown("### ⭐ Z-Score Feature (NEW!)")
        st.markdown("""
        **The Key Innovation:**
            Z-score = (driver_lap_time - circuit_avg) / circuit_std
    
    Example:
    - 95s at Monaco (avg 94s): z = +0.67 (normal) ✓
    - 95s at Monza (avg 82s): z = +6.5 (very slow) ✗
            
        **Benefits:**
        - Circuit-aware predictions
        - Better generalization
        - accuracy improvement
        """)

# ============================================================================
# TAB 3: HOW IT WORKS
# ============================================================================

with tab3:
    st.markdown("## 📈 How It Works")
    
    st.markdown("""
    ### 🧩 Prediction Process
    
    1. **Input** → First 10 laps data (lap times, position, etc.)
    2. **Z-Score** → Convert lap time to circuit-relative metric
    3. **Normalize** → Scale all features to training distribution
    4. **Predict** → Neural network predicts probability
    5. **Classify** → Compare probability to threshold
    
    ### 💡 Why Z-Scores Matter
    
    Without Z-scores: Same lap time has different meanings on different circuits
    With Z-scores: Consistent interpretation across all circuits
    
    Result: Model learns circuit context → Better predictions!
    """)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    st.markdown("## About Grid2Flag")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ℹ️ Project Info
        
        F1 race predictor using neural network
        trained from scratch with NumPy.
        
        **Data Source:**
        [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Tech Stack
        
        - ML: NumPy (from scratch!)
        - App: Streamlit
        - Data: Pandas, NumPy
        
        ### 📚 Notebooks
        1. EDA
        2. Feature Engineering
        3. Data Prep
        4. Training
        5. Evaluation
        6. Statistics
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Races", "1,125")
    with col2:
        st.metric("Drivers", "859")
    with col3:
        st.metric("Circuits", "77")
    with col4:
        st.metric("Samples", "8,468")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #999;'>🏁 Grid2Flag | Built with Streamlit</p>", unsafe_allow_html=True)
