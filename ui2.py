import streamlit as st
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Music Emotion Analyzer", layout="wide")

# -----------------------------
# MODEL
# -----------------------------
class CNN_GRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.fc(out)

@st.cache_resource
def load_models():
    model = CNN_GRU_Model()
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()

    try:
        rf_model = joblib.load("rf_model.pkl")
        rf_available = True
    except:
        rf_model = None
        rf_available = False

    return model, rf_model, rf_available

model, rf_model, rf_available = load_models()

# -----------------------------
# NORMALIZATION
# -----------------------------
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")
Y_mean = np.load("Y_mean.npy")
Y_std = np.load("Y_std.npy")

# -----------------------------
# HELPERS
# -----------------------------
def describe_emotion(val, aro):
    if val > 0 and aro > 0:
        return "Energetic and uplifting (like a party or workout track)"
    elif val > 0 and aro < 0:
        return "Calm and pleasant (like relaxing music)"
    elif val < 0 and aro > 0:
        return "Intense or tense (dramatic or aggressive)"
    else:
        return "Sad or mellow (emotional and reflective)"

def explain_frequency(top_idx):
    avg = np.mean(top_idx)
    if avg > 80:
        return "High-frequency sounds dominate → sharp, bright, energetic audio"
    elif avg > 40:
        return "Mid-frequency sounds dominate → vocals, harmony and melody drive emotion"
    else:
        return "Low-frequency sounds dominate → bass and depth influence mood"

def emotion_color(val, aro):
    if val > 0 and aro > 0:
        return "orange"
    elif val > 0 and aro < 0:
        return "green"
    elif val < 0 and aro > 0:
        return "red"
    else:
        return "blue"

# -----------------------------
# FEATURE EXTRACTION (UPDATED)
# -----------------------------
def extract_mel_chunks(y, sr, start_time, duration):
    start = int(start_time * sr)
    end = int((start_time + duration) * sr)
    y = y[start:end]

    chunk_size = int(0.5 * sr)
    features = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        if len(chunk) == chunk_size:
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel)
            features.append(mel_db.mean(axis=1))

    return np.array(features)

# -----------------------------
# UI
# -----------------------------
st.title("🎵 Music Emotion Analyzer")

sample_map = {
    "Sample 1": "samples/21.mp3",
    "Sample 2": "samples/2043.mp3",
    "Sample 3": "samples/2050.mp3",
    "Sample 4": "samples/allofme.mp3",
    "Sample 5": "samples/normal.mp3",
    "Sample 6": "samples/Come over(Prod. SUGA).mp3",
}

st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox(
    "Model",
    ["Deep Model", "Random Forest"]
)

sample_choice = st.sidebar.selectbox(
    "Or try a sample",
    ["None"] + list(sample_map.keys())
)

uploaded_file = st.sidebar.file_uploader("Upload MP3", type=["mp3"])
duration = st.sidebar.slider("Segment Length (sec)", 5, 45, 30)

# -----------------------------
# AUDIO LOADING (FIXED)
# -----------------------------
y = None
sr = None
audio_path = None

# Priority: uploaded file > sample
if uploaded_file is not None:
    audio_path = "temp.mp3"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

elif sample_choice != "None":
    audio_path = sample_map[sample_choice]

# Load audio if available
if audio_path is not None:
    st.audio(audio_path)
    y, sr = librosa.load(audio_path, sr=22050)

    max_time = max(0, int(len(y)/sr) - duration)

    start_time = st.sidebar.slider(
        "Start Time (sec)",
        0,
        max_time,
        0
    )
else:
    start_time = 0


# -----------------------------
# ANALYSIS
# -----------------------------
if st.button("Analyze") and y is not None:

    with st.spinner("Analyzing..."):

        features = extract_mel_chunks(y, sr, start_time, duration)

        features_norm = (features - X_mean) / X_std

        if model_choice == "Deep Model":
            X = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                preds = model(X).squeeze(0).numpy()
            preds = preds * Y_std + Y_mean
        else:
            preds = rf_model.predict(features)

        valence = gaussian_filter1d(preds[:,0], sigma=2)
        arousal = gaussian_filter1d(preds[:,1], sigma=2)

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2 = st.tabs(["📈 Emotion", "🧠 Explanation"])

    theme = st.get_option("theme.base")
    is_dark = theme == "dark"

    if is_dark:
        bg_color = "#0E1117"
        grid_color = "rgba(255,255,255,0.1)"
        axis_color = "white"
        line_color = "white"
        text_color = "white"
    else:
        bg_color = "white"
        grid_color = "rgba(0,0,0,0.1)"
        axis_color = "black"
        line_color = "black"
        text_color = "black"

    # -----------------------------
    # TAB 1
    # -----------------------------
    with tab1:
        st.subheader("Emotion Over Time")

        time_axis = np.arange(len(valence)) * 0.5
        color = emotion_color(valence.mean(), arousal.mean())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=valence, line=dict(color=color), name="Valence"))
        fig.add_trace(go.Scatter(x=time_axis, y=arousal, name="Arousal"))

        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Emotion Value",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        val = valence.mean()
        aro = arousal.mean()

        st.divider()

        # -----------------------------
        # SUBTABS
        # -----------------------------
        subtab1, subtab2, subtab3 = st.tabs([
            "Summary",
            "Valence and Arousal",
            "Interpretation of Results"
        ])

        # -----------------------------
        # SUMMARY
        # -----------------------------
        with subtab1:
            st.subheader("Summary of Predictions")

            col1, col2 = st.columns(2)
            col1.metric("Valence", f"{val:.2f}")
            col2.metric("Arousal", f"{aro:.2f}")

            st.write(
                "The model predicts emotional values continuously across the selected segment. "
                "Instead of assigning a single label, it produces a sequence of values that reflect how "
                "the emotional character of the music changes over time."
            )

            st.write(
                "The values shown above represent the average emotional state of the segment. "
                "These should be interpreted as relative indicators rather than exact measurements."
            )

            st.markdown("**Overall description:**")
            st.write(describe_emotion(val, aro))

        # -----------------------------
        # THEORY
        # -----------------------------
        with subtab2:
            st.subheader("Valence")

            st.write(
                "Valence measures the positivity or negativity of the music. "
                "It reflects how pleasant, bright, or emotionally uplifting a segment sounds. "
                "Higher values correspond to more positive emotional content, while lower values "
                "indicate more negative or subdued tones."
            )

            st.write(
                "In practice, valence is often influenced by factors such as harmony, melody, and timbre. "
                "For example, consonant harmonies and major tonal structures tend to produce higher valence, "
                "while dissonance or darker tonal qualities may reduce it."
            )

            st.subheader("Arousal")

            st.write(
                "Arousal captures the level of energy or intensity in the music. "
                "It reflects whether a segment feels calm and relaxed or active and energetic."
            )

            st.write(
                "Arousal is often associated with tempo, rhythm, and overall sound intensity. "
                "Faster tempos, stronger beats, and higher amplitude variations typically lead to higher arousal, "
                "while slower and softer passages tend to produce lower values."
            )

            st.subheader("Combined Representation")

            st.write(
                "Together, valence and arousal form a continuous two-dimensional representation of emotion. "
                "This allows music to be described more flexibly than discrete labels such as 'happy' or 'sad'. "
                "For example, a piece can be both low in valence and high in arousal, reflecting an intense or tense emotional state."
            )

        # -----------------------------
        # INTERPRETATION
        # -----------------------------
        with subtab3:
            st.subheader("Interpreting the Results")

            st.write(
                "The curves above represent how emotional perception changes across the selected segment. "
                "Rather than treating the song as having a single emotion, the model captures continuous variation. "
                "These changes often align with musical structure, such as transitions between sections, changes in instrumentation, "
                "or shifts in intensity."
            )

            val_range = np.max(valence) - np.min(valence)
            aro_range = np.max(arousal) - np.min(arousal)

            st.subheader("Reading the Curves")

            st.write(
                "Valence and arousal should be read together rather than in isolation. "
                "For example, a rise in arousal without a change in valence may indicate an increase in energy "
                "without a change in emotional tone. In contrast, simultaneous changes in both signals often reflect "
                "stronger emotional transitions."
            )

            st.write(
                "Sharp increases in arousal typically correspond to moments where the music becomes more intense, "
                "such as the start of a chorus, the introduction of percussion, or an increase in tempo. "
                "Gradual changes, on the other hand, often reflect smoother transitions or buildup phases."
            )

            st.write(
                "Valence tends to change more slowly. A gradual increase in valence may indicate a shift toward a more uplifting "
                "or resolved section, while a decrease may reflect a move toward tension or emotional depth."
            )

            st.subheader("Examples of Patterns")

            st.write(
                "A common pattern is a rise in arousal followed by stabilization. This often reflects a buildup and release structure, "
                "where energy increases and then settles into a steady state."
            )

            st.write(
                "Another pattern is divergence between the two signals. For instance, high arousal combined with low valence may indicate "
                "an intense but negative emotional state, such as tension or aggression. Conversely, low arousal with high valence may "
                "correspond to calm and pleasant sections."
            )

            st.write(
                "If both valence and arousal fluctuate together, this suggests strong emotional variation across the segment. "
                "This is often observed in music with clear structural contrasts."
            )

            st.subheader("Emotional Dynamics in This Segment")

            if val_range > 0.5 or aro_range > 0.5:
                st.write(
                    "The curves show noticeable variation over time. This suggests that the segment contains multiple emotional phases, "
                    "rather than a single consistent mood. These variations may align with structural changes such as transitions between "
                    "sections or changes in instrumentation."
                )
            else:
                st.write(
                    "The curves remain relatively stable, indicating a consistent emotional character throughout the segment. "
                    "This is typical of music that maintains a steady structure and mood."
                )

            st.subheader("Observed Variation")

            st.write(
                f"Valence variation: {val_range:.2f}. "
                "This reflects how much the perceived positivity changes across the segment."
            )

            st.write(
                f"Arousal variation: {aro_range:.2f}. "
                "This reflects how much the energy level changes over time."
            )

            st.subheader("Important Note")

            st.write(
                "The model captures overall trends rather than exact emotional values. "
                "Small fluctuations should not be overinterpreted, while larger patterns are more meaningful. "
                "The goal is to understand how the emotional trajectory evolves, not to assign precise emotional labels at each moment."
            )


        #----------------------------------------------------------------
        
        st.divider()

        st.markdown("## Emotion in Context")
        st.write(
            "The following view places the predicted emotion within a two-dimensional space. "
            "While the previous section shows how emotion changes over time, this representation "
            "summarizes the overall emotional position of the segment."
        )
        
        
        
        st.subheader("Emotion Map (Valence–Arousal Space)")

        st.write(
            "Each point in this space represents a combination of positivity (valence) and energy (arousal). "
            "The trajectory shows how emotion evolves across the segment. Each point corresponds to a moment in time, "
            "while the highlighted marker represents the average emotional state."
        )


        fig = go.Figure()

        # ---- CROSS AXES (GRID LIKE +) ----
        fig.add_shape(
            type="line",
            x0=-1, x1=1,
            y0=0, y1=0,
            line=dict(color="black", width=2)
        )

        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=-1, y1=1,
            line=dict(color="black", width=2)
        )

        # ---- BACKGROUND QUADRANT COLORS ----
        fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1,
                    fillcolor="rgba(255,165,0,0.15)", line=dict(width=0))  # excited

        fig.add_shape(type="rect", x0=-1, x1=0, y0=0, y1=1,
                    fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))  # angry

        fig.add_shape(type="rect", x0=0, x1=1, y0=-1, y1=0,
                    fillcolor="rgba(0,200,0,0.1)", line=dict(width=0))  # calm

        fig.add_shape(type="rect", x0=-1, x1=0, y0=-1, y1=0,
                    fillcolor="rgba(0,0,255,0.1)", line=dict(width=0))  # sad

        # ---- LABELS ----
        fig.add_annotation(x=0.7, y=0.7, text="Excited", showarrow=False)
        fig.add_annotation(x=-0.7, y=0.7, text="Angry", showarrow=False)
        fig.add_annotation(x=0.7, y=-0.7, text="Calm", showarrow=False)
        fig.add_annotation(x=-0.7, y=-0.7, text="Sad", showarrow=False)

        # ---- PREDICTION POINT ----
        # ---- TRAJECTORY (ALL TIME STEPS) ----


        # ---- START POINT ----
        fig.add_trace(go.Scatter(
            x=[valence[0]],
            y=[arousal[0]],
            mode="markers",
            marker=dict(size=12, color="green"),
            name="Start"
        ))

        # ---- END POINT ----
        fig.add_trace(go.Scatter(
            x=[valence[-1]],
            y=[arousal[-1]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="End"
        ))

        # ---- AVERAGE POINT (your original dot) ----
        fig.add_trace(go.Scatter(
            x=[val],
            y=[aro],
            mode="markers",
            marker=dict(
                size=16,
                color="blue",
                line=dict(width=2, color="white")
            ),
            name="Average"
        ))

        fig.add_trace(go.Scatter(
            x=valence,
            y=arousal,
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(
                size=5,
                color=np.arange(len(valence)),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Time",
                    len=0.6,
                    x=1.15  # 👉 move colorbar to the right
                )
            ),
            name="Trajectory"
        ))

        padding = 0.1

        x_min = max(-1, min(valence) - padding)
        x_max = min(1, max(valence) + padding)

        y_min = max(-1, min(arousal) - padding)
        y_max = min(1, max(arousal) + padding)

        # ---- LAYOUT ----
        fig.update_layout(
            plot_bgcolor=bg_color,
            xaxis=dict(
                title="Valence (Negative → Positive)",
                range=[x_min, x_max],
                zeroline=False,
                showgrid=True,
                gridcolor=grid_color,
                color=axis_color,
            ),
            yaxis=dict(
                title="Arousal (Calm → Energetic)",
                range=[y_min, y_max],
                zeroline=False,
                showgrid=True,
                gridcolor=grid_color,
                color=axis_color,
            ),
            height=450,
            showlegend=True,
            margin=dict(r=120)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "View is automatically scaled to focus on the emotional range of this segment."
        )
        

        st.markdown("""
                **How to interpret this visualization:**

                - The path shows how emotion changes over time (every 0.5 seconds).
                - The starting point (green) represents the beginning of the segment.
                - The ending point (red) represents the final emotional state.
                - The central marker indicates the overall average emotion.

                **Axes:**

                - Horizontal (Valence):  
                Moving right indicates more positive emotion, while moving left reflects more negative or tense emotion.

                - Vertical (Arousal):  
                Moving upward indicates higher energy, while moving downward reflects calmer states.

                This view helps identify whether the music stays within one emotional region or moves across multiple states.
                """)
        

        st.divider()
        st.markdown("## Distribution of Emotion")
        st.write(
            "While the map shows the average emotional position, the distributions below show how emotion is spread across the entire segment. "
            "This helps identify whether the emotion is consistent or varies significantly over time."
        )
        
        st.subheader("Emotion Distribution")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=valence,
            name="Valence",
            opacity=0.6,
            marker=dict(color="orange")
        ))

        fig.add_trace(go.Histogram(
            x=arousal,
            name="Arousal",
            opacity=0.6,
            marker=dict(color="red")
        ))

        fig.update_layout(
            barmode="overlay",
            xaxis_title="Valence / Arousal Value (−1 to 1)",
            yaxis_title="Number of Time Steps",
            legend_title="Emotion Dimension"
        )

        st.plotly_chart(fig)

        st.write(
            "A narrow distribution indicates that the emotion remains relatively stable throughout the segment. "
            "A wider spread suggests greater variation, meaning the music moves between different emotional states."
        )

        st.write(
            "For example, a wide arousal distribution may reflect alternating calm and energetic sections, "
            "while a concentrated valence distribution suggests a consistent emotional tone."
        )
        
        st.caption(
        "Note: These visualizations are complementary. The time-series shows how emotion evolves, "
        "the map shows overall position, and the distribution shows variability."
        )


    
    # TAB 2 (EXPLANATION)
    
    with tab2:
        if model_choice == "Random Forest":

            st.subheader("Understanding Model Decisions")

            st.write(
                "This section explains how the model arrives at its predictions. "
                "Instead of treating the model as a black box, we analyze which parts of the audio signal "
                "have the strongest influence on the predicted emotion."
            )

            st.divider()

            # -----------------------------
            # SHAP COMPUTATION
            # -----------------------------
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(features)

            if isinstance(shap_values, list):
                shap_val = shap_values[0]
            else:
                shap_val = shap_values

            if shap_val.ndim == 3:
                shap_val = shap_val[:, :, 0]

            importance = np.abs(shap_val).mean(axis=0)

            # -----------------------------
            # TOP FEATURES
            # -----------------------------
            st.subheader("Most Influential Frequency Bands")

            top_idx = np.argsort(importance)[-10:]
            top_vals = importance[top_idx]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=top_vals,
                y=[f"Band {i}" for i in top_idx],
                orientation="h",
                marker=dict(
                    color=top_vals,
                    colorscale="Viridis"
                )
            ))

            fig.update_layout(
                xaxis_title="Importance",
                yaxis_title="Mel Frequency Band",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write(
                "Each bar represents a frequency band from the mel spectrogram. "
                "Higher values indicate that the model relies more heavily on that band when predicting emotion."
            )

            st.write(
                "These bands correspond to ranges of frequencies in the audio signal. "
                "Instead of manually defining features like tempo or pitch, the model learns which frequency regions matter directly from the data."
            )

            st.divider()

            # -----------------------------
            # GROUPED INTERPRETATION
            # -----------------------------
            st.subheader("Frequency Region Analysis")

            low = importance[:40].mean()
            mid = importance[40:80].mean()
            high = importance[80:].mean()

            labels = ["Low (Bass)", "Mid (Vocals/Melody)", "High (Timbre/Texture)"]
            values = [low, mid, high]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker=dict(
                    color=["#1f77b4", "#2ca02c", "#ff7f0e"]
                )
            ))

            fig.update_layout(
                yaxis_title="Average Importance",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write(
                "To make the results easier to interpret, the frequency bands are grouped into three broad regions. "
                "Low frequencies capture bass and rhythm, mid frequencies capture melodic and harmonic structure, which may come from vocals or instruments depending on the track, and high frequencies capture sharp or bright sounds."
            )

            freq_explanation = explain_frequency(top_idx)

            st.write(
                f"The model is primarily influenced by: {freq_explanation.lower()}."
            )

            st.write(
                "This provides insight into how different parts of the audio contribute to emotional perception. "
                "For example, strong bass patterns may contribute to calm or steady emotions, while high-frequency components "
                "may increase perceived intensity."
            )

            st.divider()

            # -----------------------------
            # CONNECTION TO EMOTION
            # -----------------------------
            st.subheader("Connection to Predicted Emotion")

            val = valence.mean()
            aro = arousal.mean()

            st.write(
                "The importance patterns can be linked back to the predicted emotional state. "
                "Different frequency regions tend to influence valence and arousal in different ways."
            )

            st.markdown(f"**Predicted emotion:** {describe_emotion(val, aro)}")

            st.write(
                "For instance, higher energy (arousal) is often associated with increased activity in higher frequencies, "
                "while valence is more closely related to harmonic and melodic content captured in mid-frequency bands."
            )

            st.write(
                "By combining these observations, we can better understand not just what the model predicts, "
                "but why it makes those predictions."
            )

            st.divider()

            # -----------------------------
            # MODEL CONFIDENCE
            # -----------------------------
            st.subheader("Model Behavior")

            variation = np.var(valence + arousal)

            st.write(
                f"The variation in predictions across time is {variation:.3f}. "
                "Higher variation suggests that the model detects meaningful changes in the audio, "
                "while lower variation indicates a more stable emotional structure."
            )

            st.caption(
                "These explanations are approximate and reflect patterns learned by the model rather than direct causal relationships."
            )

        else:
            st.info("Use Random Forest model for interpretability and feature analysis.")

        




    

else:
    st.info("Upload a song to begin")