
import os
import shutil
import tempfile
import warnings

import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from app.src.deepfake import infa_deepfake  # your existing inference function

# Hide TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_audio_metadata(filepath):
    """Extract details about the audio using soundfile instead of audioread."""
    y, sr = sf.read(filepath, always_2d=False)

    # If stereo, convert to mono
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    # Ensure numpy array
    y = np.array(y, dtype=np.float32)

    # Resample to 16k if needed
    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration = librosa.get_duration(y=y, sr=sr)

    # Safe pitch / energy estimation
    try:
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = float(np.nanmean(pitch))
    except Exception:
        avg_pitch = float("nan")

    try:
        rms = librosa.feature.rms(y=y)
        avg_energy = float(np.nanmean(rms))
    except Exception:
        avg_energy = float("nan")

    return {
        "samples": len(y),
        "sr": sr,
        "duration": duration,
        "avg_pitch": avg_pitch,
        "avg_energy": avg_energy,
        "waveform": y,
    }


def process_audio_file(uploaded_file):
    """
    Common processing for:
      - Uploaded file (st.file_uploader)
      - Recorded audio (st.audio_input)
    Steps:
      - Save to temp file
      - Extract metadata
      - Call infa_deepfake
      - Cleanup
    """
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    print("filepath: ", temp_file_path)
    
    try:
        data = uploaded_file.read()

        with open(temp_file_path, "wb") as f:
            f.write(data)

        # Try metadata, but don't fail inference if it breaks
        try:
            audio_info = extract_audio_metadata(temp_file_path)
        except Exception as e:
            print("Metadata extraction failed:", e)
            audio_info = {
                "samples": 0,
                "sr": 0,
                "duration": 0.0,
                "avg_pitch": float("nan"),
                "avg_energy": float("nan"),
                "waveform": np.array([]),
            }

        status, message = infa_deepfake(temp_file_path)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return status, message, audio_info


def render_results(status, message, info):
    """Render model result + metadata in a modern card layout."""
    st.markdown(
        """
        <div class="section-title">
            <span class="pill pill-primary">Analysis</span>
            <h2>🧪 Detection Result</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # RESULT CARD
    if status == 1:
        is_fake = "fake" in str(message).lower()
        if is_fake:
            bg = "#fee2e2"
            border = "#fecaca"
            heading_color = "#b91c1c"
            text_color = "#7f1d1d"
            title = "❌ Deepfake Detected"
        else:
            bg = "#dcfce7"
            border = "#bbf7d0"
            heading_color = "#166534"
            text_color = "#065f46"
            title = "✅ Real Audio"

        st.markdown(
            f"""
            <div class="card result-card" style="
                background:{bg};
                border:1px solid {border};
            ">
                <h3 style="color:{heading_color};margin-bottom:0.5rem;">{title}</h3>
                <p style="color:{text_color};font-size:1rem;margin-top:0.25rem;">
                    {message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="card result-card" style="
                background:#fef9c3;
                border:1px solid #fef08a;
            ">
                <h3 style="color:#854d0e;margin-bottom:0.5rem;">⚠️ Inference Failed</h3>
                <p style="color:#713f12;font-size:1rem;margin-top:0.25rem;">
                    Something went wrong while processing this audio.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(message)

    st.markdown("<br>", unsafe_allow_html=True)

    # METADATA + SPEAKER INFO CARDS
    st.markdown(
        """
        <div class="section-title">
            <span class="pill pill-secondary">Audio profile</span>
            <h2>🎛️ Audio & Speaker Characteristics</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Duration</div>
                <div class="card-value">{info['duration']:.2f}<span class="card-unit"> sec</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Sample Rate</div>
                <div class="card-value">{info['sr']}<span class="card-unit"> Hz</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Samples</div>
                <div class="card-value">{info['samples']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Average Pitch</div>
                <div class="card-value">
                    {"" if np.isnan(info['avg_pitch']) else f"{info['avg_pitch']:.2f}"}
                    <span class="card-unit"> Hz</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Voice Energy</div>
                <div class="card-value">
                    {"" if np.isnan(info['avg_energy']) else f"{info['avg_energy']:.5f}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ These are **acoustic characteristics**, not speaker identity "
        "(they describe how the voice sounds, not who it is)."
    )

    # WAVEFORM
    st.markdown(
        """
        <div class="section-title">
            <span class="pill pill-outline">Signal view</span>
            <h2>📈 Waveform</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if info["waveform"].size > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(info["waveform"])
        ax.set_title("Waveform", fontsize=11)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    else:
        st.info("Waveform not available for this audio.")


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="DeepFake Voice Detection",
        page_icon="🎙️",
        layout="wide",
    )

    # --- session state for mode and result ---
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"  # 'upload' or 'record'

    if "result" not in st.session_state:
        st.session_state.result = {
            "has_result": False,
            "status": None,
            "message": None,
            "info": None,
        }

    # Global CSS theme
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1f2937 0, #020617 40%, #020617 100%);
            color: #e5e7eb;
        }
        .card {
            padding: 1.1rem 1.3rem;
            border-radius: 1rem;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
        }
        .small-card {
            padding: 0.9rem 1.0rem;
        }
        .card-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #9ca3af;
            margin-bottom: 0.05rem;
        }
        .card-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #e5e7eb;
        }
        .card-unit {
            font-size: 0.8rem;
            margin-left: 0.25rem;
            color: #9ca3af;
        }
        .section-title h2 {
            margin-bottom: 0.2rem;
        }
        .section-title {
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .pill-primary {
            background: rgba(59, 130, 246, 0.12);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.5);
        }
        .pill-secondary {
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .pill-outline {
            background: transparent;
            color: #e5e7eb;
            border: 1px dashed rgba(148, 163, 184, 0.7);
        }
        /* Mode cards */
        .mode-card {
            cursor: pointer;
            transition: all 0.18s ease-out;
            border-radius: 0.9rem;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(148,163,184,0.35);
            background: rgba(15,23,42,0.8);
        }
        .mode-card:hover {
            border-color: #60a5fa;
            box-shadow: 0 0 0 1px rgba(37,99,235,0.5);
            transform: translateY(-1px);
        }
        .mode-card.active {
            border-color: #60a5fa;
            background: radial-gradient(circle at top left, rgba(37,99,235,0.35), rgba(15,23,42,0.9));
        }
        .mode-title {
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .mode-desc {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # HERO
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1.8rem;">
            <div style="
                display:inline-flex;
                padding:0.25rem 0.65rem;
                border-radius:999px;
                background:rgba(15,23,42,0.85);
                border:1px solid rgba(148,163,184,0.4);
                font-size:0.75rem;
                letter-spacing:0.16em;
                text-transform:uppercase;
                color:#9ca3af;
                margin-bottom:0.6rem;
            ">
                Deepfake Voice Intelligence
            </div>
            <h1 style="color:#e5e7eb;font-size:2.2rem;margin-bottom:0.3rem;">
                🎙️ DeepFake Voice Detection
            </h1>
            <p style="color:#9ca3af;font-size:0.95rem;max-width:620px;margin:0 auto;">
                A YAMNet-based pipeline that analyzes speech, extracts acoustic features, and predicts whether a voice clip
                is <b>real</b> or <b>deepfake</b>. Use the <b>Input Panel</b> on the left and view results in the <b>Analysis Panel</b> on the right.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- MAIN TWO-COLUMN LAYOUT (left = Input Panel, right = Analysis Panel) ---
    left_col, right_col = st.columns(2)

    # ---------------- LEFT COLUMN: INPUT PANEL ----------------
    with left_col:
        # Put the whole left side inside a bordered box
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-primary">Input panel</span>
                    <h2>🎚️ Provide audio to analyze</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- MODE TOGGLE: buttons row ---
        mode_btn_cols = st.columns(2)

        # 1️⃣ Capture button clicks
        with mode_btn_cols[0]:
            upload_clicked = st.button(
                "📤 Upload file",
                key="mode_upload_btn",
                use_container_width=True,
            )

        with mode_btn_cols[1]:
            record_clicked = st.button(
                "🎙️ Record now",
                key="mode_record_btn",
                use_container_width=True,
            )

        # 2️⃣ Update mode BEFORE rendering cards
        if upload_clicked:
            st.session_state.mode = "upload"
        elif record_clicked:
            st.session_state.mode = "record"

        st.markdown("<br>", unsafe_allow_html=True)

        # --- MODE CARDS ROW (uses final mode value) ---
        mode_card_cols = st.columns(2)

        with mode_card_cols[0]:
            st.markdown(
                f"""
                <div class="mode-card {'active' if st.session_state.mode=='upload' else ''}">
                    <div class="mode-title">Upload audio</div>
                    <div class="mode-desc">
                        Drag &amp; drop an existing clip from your system.
                        Works best with clean speech segments.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with mode_card_cols[1]:
            st.markdown(
                f"""
                <div class="mode-card {'active' if st.session_state.mode=='record' else ''}">
                    <div class="mode-title">Record from mic</div>
                    <div class="mode-desc">
                        Use your microphone to capture a short sample directly
                        in the browser and analyze it instantly.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Input controls based on selected mode ---
        if st.session_state.mode == "upload":
            # FULL-WIDTH upload section (aligned like before)
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-secondary">Upload</span>
                    <h3>📂 Select an audio file</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                st.markdown("#### 🔊 Preview")
                st.audio(uploaded_file.getvalue(), format="audio/wav")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button(
                    "🔍 Run analysis",
                    key="analyze_upload",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing uploaded audio..."):
                        uploaded_file.seek(0)
                        status, message, info = process_audio_file(uploaded_file)

                    st.session_state.result = {
                        "has_result": True,
                        "status": status,
                        "message": message,
                        "info": info,
                    }

        else:  # record mode
            # FULL-WIDTH record section (aligned like the screenshot you liked)
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-secondary">Record</span>
                    <h3>🎤 Capture from microphone</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            recorded_audio = st.audio_input(
                "Click the mic icon to start recording",
                sample_rate=16000,
                key="recorder",
            )

            if recorded_audio is not None:
                st.markdown("#### 🔊 Preview")
                st.audio(recorded_audio, format="audio/wav")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button(
                    "🔍 Run analysis",
                    key="analyze_record",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing recorded audio..."):
                        recorded_audio.seek(0)
                        status, message, info = process_audio_file(recorded_audio)

                    st.session_state.result = {
                        "has_result": True,
                        "status": status,
                        "message": message,
                        "info": info,
                    }


    # ---------------- RIGHT COLUMN: ANALYSIS PANEL ----------------
    with right_col:
        # Wrap analysis side in its own box
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-outline">Analysis panel</span>
                    <h2>📊 Deepfake verdict & signal view</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.result["has_result"]:
                render_results(
                    st.session_state.result["status"],
                    st.session_state.result["message"],
                    st.session_state.result["info"],
                )
            else:
                # Placeholder card when nothing analyzed yet
                st.markdown(
                    """
                    <div class="card" style="margin-top:0.6rem;">
                        <h3 style="margin-bottom:0.4rem;">Awaiting audio input</h3>
                        <p style="color:#9ca3af;font-size:0.9rem;margin-bottom:0.6rem;">
                            Use the <b>Input Panel</b> on the left to upload a clip or record from your microphone,
                            then click <b>Run analysis</b>. The detection result and signal insights will appear here.
                        </p>
                        <ul style="color:#9ca3af;font-size:0.85rem;line-height:1.5;">
                            <li>Use clear speech segments of 2–10 seconds.</li>
                            <li>Avoid heavy background noise for more reliable predictions.</li>
                            <li>You can re-run analysis with different clips anytime.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
