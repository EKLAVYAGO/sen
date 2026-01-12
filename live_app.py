import streamlit as st
import requests
import time
from audio_recorder_streamlit import audio_recorder
import Levenshtein
import string
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# Practice paragraphs
PARAGRAPHS = [
    {
        "id": 1,
        "title": "Technology in Education",
        "text": "Technology has revolutionized the way we learn and teach. Online platforms provide access to educational resources from anywhere in the world. Students can now collaborate with peers across different countries and learn at their own pace. This digital transformation has made education more accessible and personalized than ever before."
    },
    {
        "id": 2,
        "title": "Climate Change",
        "text": "Climate change is one of the most pressing issues facing our planet today. Rising temperatures are causing glaciers to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. It is crucial that we take immediate action to reduce carbon emissions and protect our environment for future generations."
    },
    {
        "id": 3,
        "title": "Importance of Exercise",
        "text": "Regular physical exercise is essential for maintaining good health. It strengthens the cardiovascular system, improves mental well-being, and helps control weight. Even thirty minutes of moderate activity each day can significantly reduce the risk of chronic diseases. Making exercise a daily habit is one of the best investments you can make in your health."
    },
    {
        "id": 4,
        "title": "Artificial Intelligence",
        "text": "Artificial intelligence is transforming industries across the globe. From healthcare to finance, AI systems are helping professionals make better decisions faster. Machine learning algorithms can analyze vast amounts of data to identify patterns that humans might miss. As this technology continues to evolve, it will create new opportunities and challenges for society."
    },
    {
        "id": 5,
        "title": "The Power of Reading",
        "text": "Reading is a powerful tool for personal growth and development. Books expose us to new ideas, perspectives, and experiences beyond our own lives. They improve vocabulary, enhance critical thinking skills, and stimulate imagination. Whether fiction or non-fiction, regular reading enriches the mind and broadens our understanding of the world."
    }
]

# Filler words to detect
FILLER_WORDS = ["um", "uh", "like", "you know", "actually", "basically", "literally", "so", "well"]


# ============================================================
# ASSEMBLYAI TRANSCRIPTION FUNCTIONS
# ============================================================

def upload_audio_to_assemblyai(audio_bytes, api_key):
    """Upload audio file to AssemblyAI and get upload URL"""
    headers = {"authorization": api_key}

    response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        data=audio_bytes
    )

    if response.status_code == 200:
        return response.json()["upload_url"]
    else:
        raise Exception(f"Upload failed: {response.text}")


def transcribe_audio(upload_url, api_key):
    """Submit transcription request to AssemblyAI"""
    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    data = {
        "audio_url": upload_url
    }

    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=data,
        headers=headers
    )

    if response.status_code == 200:
        return response.json()["id"]
    else:
        raise Exception(f"Transcription request failed: {response.text}")


def get_transcription_result(transcript_id, api_key):
    """Poll AssemblyAI for transcription result"""
    headers = {"authorization": api_key}

    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        response = requests.get(url, headers=headers)
        result = response.json()

        status = result["status"]

        if status == "completed":
            return result["text"]
        elif status == "error":
            raise Exception(f"Transcription failed: {result.get('error', 'Unknown error')}")
        else:
            # Still processing, wait 2 seconds and try again
            time.sleep(2)


def transcribe_audio_file(audio_bytes, api_key):
    """Complete transcription pipeline"""
    try:
        # Step 1: Upload audio
        upload_url = upload_audio_to_assemblyai(audio_bytes, api_key)

        # Step 2: Submit transcription request
        transcript_id = transcribe_audio(upload_url, api_key)

        # Step 3: Get result
        transcript_text = get_transcription_result(transcript_id, api_key)

        return transcript_text
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def normalize_text(text):
    """Normalize text for comparison"""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def calculate_accuracy(original, transcript):
    """Calculate word-level accuracy"""
    original_words = normalize_text(original).split()
    transcript_words = normalize_text(transcript).split()

    # Use Levenshtein distance at word level
    distance = Levenshtein.distance(original_words, transcript_words)
    max_length = max(len(original_words), len(transcript_words))

    if max_length == 0:
        return 0.0

    accuracy = (1 - distance / max_length) * 100
    return max(0, accuracy)


def analyze_clarity(original, transcript):
    """Analyze clarity by finding missing and extra words"""
    original_words = set(normalize_text(original).split())
    transcript_words_list = normalize_text(transcript).split()
    transcript_words_set = set(transcript_words_list)

    missing_words = original_words - transcript_words_set
    extra_words = transcript_words_set - original_words

    return {
        "missing_words": list(missing_words),
        "extra_words": list(extra_words),
        "missing_count": len(missing_words),
        "extra_count": len(extra_words)
    }


def calculate_wpm(text, duration_seconds):
    """Calculate words per minute"""
    word_count = len(text.split())
    if duration_seconds == 0:
        return 0
    wpm = (word_count / duration_seconds) * 60
    return wpm


def detect_filler_words(transcript):
    """Detect filler words in transcript"""
    transcript_lower = normalize_text(transcript)
    words = transcript_lower.split()

    filler_count = {}
    total_fillers = 0

    for filler in FILLER_WORDS:
        count = words.count(filler)
        if count > 0:
            filler_count[filler] = count
            total_fillers += count

    return {
        "filler_words": filler_count,
        "total_fillers": total_fillers
    }


def evaluate_reading(original, transcript, duration_seconds):
    """Complete evaluation of reading performance"""

    # Accuracy
    accuracy = calculate_accuracy(original, transcript)

    # Clarity analysis
    clarity = analyze_clarity(original, transcript)
    clarity_score = 100 - (clarity["missing_count"] + clarity["extra_count"]) * 5
    clarity_score = max(0, min(100, clarity_score))

    # WPM calculation
    wpm = calculate_wpm(transcript, duration_seconds)

    # Filler words
    fillers = detect_filler_words(transcript)

    # Confidence score (based on WPM and filler words)
    # Ideal WPM is 130-170 for clear speech
    if wpm < 100:
        wpm_score = 50 + (wpm / 100) * 30  # Slow speech
    elif wpm <= 170:
        wpm_score = 80 + ((170 - abs(wpm - 150)) / 170) * 20
    else:
        wpm_score = 80 - ((wpm - 170) / 50) * 30  # Too fast

    wpm_score = max(0, min(100, wpm_score))

    # Reduce confidence for filler words
    filler_penalty = min(fillers["total_fillers"] * 5, 40)
    confidence_score = max(0, wpm_score - filler_penalty)

    return {
        "accuracy": accuracy,
        "clarity_score": clarity_score,
        "clarity_details": clarity,
        "confidence_score": confidence_score,
        "wpm": wpm,
        "fillers": fillers
    }


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Read-Aloud Practice App",
        page_icon="🎤",
        layout="wide"
    )

    st.title("🎤 Read-Aloud Interview & Speaking Practice App")
    st.markdown("Practice your reading skills and get instant feedback!")

    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Settings")

        # Initialize API key in session state if not exists
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""

        api_key = st.text_input(
            "AssemblyAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Get your free API key at https://www.assemblyai.com/"
        )

        # Update session state
        st.session_state.api_key = api_key

        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown("""
        1. Enter your API key above
        2. Read the paragraph shown
        3. Click to start recording
        4. Read aloud clearly
        5. Click to stop recording
        6. Click 'Analyze' for feedback
        """)

        st.markdown("---")
        st.info("💡 **Get your free API key:**\n\nhttps://www.assemblyai.com/")

        if st.button("🔄 Get New Paragraph"):
            if "current_paragraph" in st.session_state:
                del st.session_state.current_paragraph
            st.rerun()

    # Initialize session state
    if "current_paragraph" not in st.session_state:
        import random
        st.session_state.current_paragraph = random.choice(PARAGRAPHS)

    # Display paragraph
    paragraph = st.session_state.current_paragraph

    st.header(f"📄 {paragraph['title']}")
    st.info(paragraph["text"])

    st.markdown("---")

    # Audio recording
    st.subheader("🎙️ Record Your Reading")

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x"
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Estimate duration (approximate based on bytes)
        # This is a rough estimate - actual duration may vary
        estimated_duration = len(audio_bytes) / 16000  # Assuming 16kHz sample rate

        if st.button("📊 Analyze My Reading", type="primary"):
            # Use the API key from session state
            current_api_key = st.session_state.api_key

            if not current_api_key or current_api_key.strip() == "":
                st.error("⚠️ Please enter your AssemblyAI API key in the sidebar!")
                st.info("Get your free API key at: https://www.assemblyai.com/")
            else:
                with st.spinner("🔄 Transcribing your audio..."):
                    try:
                        # Transcribe audio - pass API key as parameter
                        transcript = transcribe_audio_file(audio_bytes, current_api_key)

                        if not transcript or transcript.strip() == "":
                            st.warning("⚠️ No speech detected. Please try recording again.")
                        else:
                            # ============================================================
                            # SHOW TRANSCRIPT FIRST
                            # ============================================================
                            st.markdown("---")
                            st.header("📝 Your Transcript")

                            col_t1, col_t2 = st.columns(2)

                            with col_t1:
                                st.subheader("🎤 What You Said")
                                st.success(transcript)

                            with col_t2:
                                st.subheader("📖 Original Text")
                                st.info(paragraph["text"])

                            # ============================================================
                            # CALCULATE EVALUATION METRICS
                            # ============================================================

                            # Evaluate reading
                            results = evaluate_reading(
                                paragraph["text"],
                                transcript,
                                estimated_duration
                            )

                            # Calculate overall score
                            overall_score = (results['accuracy'] + results['clarity_score'] + results[
                                'confidence_score']) / 3

                            # ============================================================
                            # COMPREHENSIVE ANALYSIS DASHBOARD WITH CHARTS
                            # ============================================================

                            st.markdown("---")
                            st.markdown("# 📊 Performance Dashboard")

                            # Overall Performance Banner
                            st.markdown("## 🎯 Overall Performance")

                            if overall_score >= 80:
                                st.success(f"### 🌟 Excellent! Overall Score: {overall_score:.1f}%")
                                performance_color = "#28a745"
                                performance_emoji = "🌟"
                                performance_status = "Excellent"
                            elif overall_score >= 60:
                                st.info(f"### 👍 Good Job! Overall Score: {overall_score:.1f}%")
                                performance_color = "#17a2b8"
                                performance_emoji = "👍"
                                performance_status = "Good"
                            else:
                                st.warning(f"### 📚 Keep Practicing! Overall Score: {overall_score:.1f}%")
                                performance_color = "#ffc107"
                                performance_emoji = "📚"
                                performance_status = "Needs Practice"

                            # Progress bar for overall score
                            st.progress(overall_score / 100)

                            st.markdown("---")

                            # ============================================================
                            # CHART 1: GAUGE CHART FOR OVERALL SCORE
                            # ============================================================
                            st.markdown("## 📈 Score Visualization")

                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=overall_score,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Overall Performance Score"},
                                delta={'reference': 80, 'increasing': {'color': "green"}},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': performance_color},
                                    'steps': [
                                        {'range': [0, 60], 'color': "lightgray"},
                                        {'range': [60, 80], 'color': "lightblue"},
                                        {'range': [80, 100], 'color': "lightgreen"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))

                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)

                            st.markdown("---")

                            # ============================================================
                            # CHART 2: BAR CHART FOR CORE METRICS
                            # ============================================================
                            st.markdown("## 📊 Core Performance Metrics")

                            metrics_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'Clarity', 'Confidence'],
                                'Score': [results['accuracy'], results['clarity_score'], results['confidence_score']],
                                'Color': ['#FF6B6B', '#4ECDC4', '#45B7D1']
                            })

                            fig_bar = px.bar(
                                metrics_df,
                                x='Metric',
                                y='Score',
                                color='Metric',
                                color_discrete_map={
                                    'Accuracy': '#FF6B6B',
                                    'Clarity': '#4ECDC4',
                                    'Confidence': '#45B7D1'
                                },
                                title="Performance Breakdown",
                                labels={'Score': 'Score (%)'},
                                text='Score'
                            )

                            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig_bar.update_layout(
                                showlegend=False,
                                yaxis_range=[0, 105],
                                height=400
                            )

                            st.plotly_chart(fig_bar, use_container_width=True)

                            # Show detailed metrics below chart
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    label="🎯 Accuracy",
                                    value=f"{results['accuracy']:.1f}%",
                                    help="How closely your words matched the original text"
                                )

                            with col2:
                                st.metric(
                                    label="✨ Clarity",
                                    value=f"{results['clarity_score']:.1f}%",
                                    help="Based on missing and extra words"
                                )

                            with col3:
                                st.metric(
                                    label="💪 Confidence",
                                    value=f"{results['confidence_score']:.1f}%",
                                    help="Based on speaking pace and filler words"
                                )

                            st.markdown("---")

                            # ============================================================
                            # CHART 3: RADAR CHART FOR MULTI-DIMENSIONAL VIEW
                            # ============================================================
                            st.markdown("## 🎯 Performance Radar")

                            categories = ['Accuracy', 'Clarity', 'Confidence',
                                          'Speed', 'Fluency']

                            # Calculate speed score (based on WPM)
                            if 130 <= results['wpm'] <= 170:
                                speed_score = 100
                            elif results['wpm'] < 130:
                                speed_score = (results['wpm'] / 130) * 100
                            else:
                                speed_score = max(0, 100 - ((results['wpm'] - 170) / 2))

                            # Calculate fluency score (inverse of filler words)
                            fluency_score = max(0, 100 - (results['fillers']['total_fillers'] * 10))

                            values = [
                                results['accuracy'],
                                results['clarity_score'],
                                results['confidence_score'],
                                speed_score,
                                fluency_score
                            ]

                            fig_radar = go.Figure(data=go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                fillcolor='rgba(68, 189, 209, 0.3)',
                                line=dict(color='#45B7D1', width=2)
                            ))

                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100]
                                    )
                                ),
                                showlegend=False,
                                title="Multi-Dimensional Performance View",
                                height=500
                            )

                            st.plotly_chart(fig_radar, use_container_width=True)

                            st.markdown("---")

                            # ============================================================
                            # CHART 4: DONUT CHART FOR ERROR BREAKDOWN
                            # ============================================================
                            st.markdown("## 🔍 Error Analysis")

                            col_chart1, col_chart2 = st.columns(2)

                            with col_chart1:
                                # Words accuracy breakdown
                                original_word_count = len(normalize_text(paragraph["text"]).split())
                                correct_words = original_word_count - results['clarity_details']['missing_count']

                                fig_donut1 = go.Figure(data=[go.Pie(
                                    labels=['Correct Words', 'Missing Words'],
                                    values=[correct_words, results['clarity_details']['missing_count']],
                                    hole=.4,
                                    marker_colors=['#28a745', '#dc3545']
                                )])

                                fig_donut1.update_layout(
                                    title="Words Coverage",
                                    annotations=[dict(text=f'{(correct_words / original_word_count * 100):.0f}%',
                                                      x=0.5, y=0.5, font_size=20, showarrow=False)],
                                    height=300
                                )

                                st.plotly_chart(fig_donut1, use_container_width=True)

                            with col_chart2:
                                # Filler words vs clean words
                                transcript_word_count = len(transcript.split())
                                clean_words = transcript_word_count - results['fillers']['total_fillers']

                                fig_donut2 = go.Figure(data=[go.Pie(
                                    labels=['Clean Speech', 'Filler Words'],
                                    values=[clean_words, results['fillers']['total_fillers']],
                                    hole=.4,
                                    marker_colors=['#17a2b8', '#ffc107']
                                )])

                                fig_donut2.update_layout(
                                    title="Speech Quality",
                                    annotations=[dict(text=f'{(clean_words / transcript_word_count * 100):.0f}%',
                                                      x=0.5, y=0.5, font_size=20, showarrow=False)],
                                    height=300
                                )

                                st.plotly_chart(fig_donut2, use_container_width=True)

                            st.markdown("---")

                            # ============================================================
                            # CHART 5: WPM SPEEDOMETER
                            # ============================================================
                            st.markdown("## ⏱️ Speaking Speed Analysis")

                            fig_wpm = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=results['wpm'],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Words Per Minute"},
                                gauge={
                                    'axis': {'range': [0, 250]},
                                    'bar': {'color': "#4ECDC4"},
                                    'steps': [
                                        {'range': [0, 100], 'color': "lightcoral"},
                                        {'range': [100, 130], 'color': "lightyellow"},
                                        {'range': [130, 170], 'color': "lightgreen"},
                                        {'range': [170, 200], 'color': "lightyellow"},
                                        {'range': [200, 250], 'color': "lightcoral"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "green", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 150
                                    }
                                }
                            ))

                            fig_wpm.update_layout(height=300)
                            st.plotly_chart(fig_wpm, use_container_width=True)

                            col_wpm1, col_wpm2, col_wpm3 = st.columns(3)

                            with col_wpm1:
                                st.metric("Current WPM", f"{results['wpm']:.0f}")
                            with col_wpm2:
                                st.metric("Ideal Range", "130-170")
                            with col_wpm3:
                                if results['wpm'] < 100:
                                    st.metric("Status", "🐌 Too Slow")
                                elif results['wpm'] > 180:
                                    st.metric("Status", "🚀 Too Fast")
                                else:
                                    st.metric("Status", "✅ Perfect")

                            st.markdown("---")

                            # ============================================================
                            # CHART 6: FILLER WORDS BAR CHART
                            # ============================================================
                            if results['fillers']['total_fillers'] > 0:
                                st.markdown("## 🚫 Filler Words Breakdown")

                                filler_df = pd.DataFrame({
                                    'Filler Word': list(results['fillers']['filler_words'].keys()),
                                    'Count': list(results['fillers']['filler_words'].values())
                                })

                                fig_filler = px.bar(
                                    filler_df,
                                    x='Filler Word',
                                    y='Count',
                                    color='Count',
                                    color_continuous_scale='Reds',
                                    title="Filler Words Usage",
                                    text='Count'
                                )

                                fig_filler.update_traces(textposition='outside')
                                fig_filler.update_layout(showlegend=False, height=350)

                                st.plotly_chart(fig_filler, use_container_width=True)

                            st.markdown("---")

                            # ============================================================
                            # DETAILED TEXT ANALYSIS
                            # ============================================================
                            st.markdown("## 📝 Detailed Word Analysis")

                            error_col1, error_col2 = st.columns(2)

                            with error_col1:
                                st.markdown("### ❌ Missing Words")
                                if results['clarity_details']['missing_count'] > 0:
                                    st.error(f"**Count:** {results['clarity_details']['missing_count']}")
                                    missing_display = ', '.join(results['clarity_details']['missing_words'][:20])
                                    st.write(missing_display)
                                    if results['clarity_details']['missing_count'] > 20:
                                        st.caption(f"... and {results['clarity_details']['missing_count'] - 20} more")
                                else:
                                    st.success("✅ No missing words!")

                            with error_col2:
                                st.markdown("### ➕ Extra Words")
                                if results['clarity_details']['extra_count'] > 0:
                                    st.warning(f"**Count:** {results['clarity_details']['extra_count']}")
                                    extra_display = ', '.join(results['clarity_details']['extra_words'][:20])
                                    st.write(extra_display)
                                    if results['clarity_details']['extra_count'] > 20:
                                        st.caption(f"... and {results['clarity_details']['extra_count'] - 20} more")
                                else:
                                    st.success("✅ No extra words!")

                            st.markdown("---")

                            # ============================================================
                            # PERSONALIZED IMPROVEMENT TIPS
                            # ============================================================
                            st.markdown("## 💡 Personalized Improvement Tips")

                            tips = []

                            # Accuracy tips
                            if results['accuracy'] < 70:
                                tips.append(
                                    "📖 **Focus on accuracy**: Try reading more slowly and clearly to match the text better.")
                            elif results['accuracy'] < 90:
                                tips.append("🎯 **Good accuracy**: Minor improvements will get you to excellence!")
                            else:
                                tips.append("🌟 **Excellent accuracy**: Your reading matches the text very well!")

                            # WPM tips
                            if results['wpm'] < 100:
                                tips.append(
                                    "⏩ **Speed up**: Try to speak a bit faster for more natural delivery (aim for 130-170 WPM).")
                            elif results['wpm'] > 180:
                                tips.append(
                                    "🐢 **Slow down**: Speaking too fast can reduce clarity (aim for 130-170 WPM).")
                            else:
                                tips.append("✅ **Perfect pace**: Your speaking speed is in the ideal range!")

                            # Filler word tips
                            if results['fillers']['total_fillers'] > 5:
                                tips.append(
                                    "🚫 **Reduce fillers**: Practice pausing instead of using filler words like 'um' and 'uh'.")
                            elif results['fillers']['total_fillers'] > 0:
                                tips.append(
                                    "👍 **Minimal fillers**: You're doing well, keep working on eliminating those few fillers!")
                            else:
                                tips.append("🌟 **Zero fillers**: Excellent! You spoke without any filler words!")

                            # Missing/extra words tips
                            if results['clarity_details']['missing_count'] > 5:
                                tips.append(
                                    "📚 **Complete the text**: You missed several words. Try to read the entire paragraph.")
                            if results['clarity_details']['extra_count'] > 5:
                                tips.append(
                                    "🎯 **Stick to the script**: You added extra words. Focus on reading exactly what's written.")

                            # Display tips in columns
                            tip_cols = st.columns(2)
                            for idx, tip in enumerate(tips):
                                with tip_cols[idx % 2]:
                                    st.info(tip)

                            st.markdown("---")

                            # ============================================================
                            # PERFORMANCE SUMMARY TABLE
                            # ============================================================
                            st.markdown("## 📋 Complete Performance Summary")

                            summary_data = {
                                "Metric": [
                                    "Overall Score",
                                    "Accuracy",
                                    "Clarity",
                                    "Confidence",
                                    "Words Per Minute",
                                    "Filler Words",
                                    "Missing Words",
                                    "Extra Words",
                                    "Words Spoken"
                                ],
                                "Value": [
                                    f"{overall_score:.1f}%",
                                    f"{results['accuracy']:.1f}%",
                                    f"{results['clarity_score']:.1f}%",
                                    f"{results['confidence_score']:.1f}%",
                                    f"{results['wpm']:.0f}",
                                    str(results['fillers']['total_fillers']),
                                    str(results['clarity_details']['missing_count']),
                                    str(results['clarity_details']['extra_count']),
                                    f"{len(transcript.split())}/{len(paragraph['text'].split())}"
                                ],
                                "Status": [
                                    performance_emoji,
                                    "✅" if results['accuracy'] >= 80 else "⚠️" if results['accuracy'] >= 60 else "❌",
                                    "✅" if results['clarity_score'] >= 80 else "⚠️" if results[
                                                                                           'clarity_score'] >= 60 else "❌",
                                    "✅" if results['confidence_score'] >= 80 else "⚠️" if results[
                                                                                              'confidence_score'] >= 60 else "❌",
                                    "✅" if 130 <= results['wpm'] <= 170 else "⚠️",
                                    "✅" if results['fillers']['total_fillers'] == 0 else "⚠️" if results['fillers'][
                                                                                                     'total_fillers'] <= 3 else "❌",
                                    "✅" if results['clarity_details']['missing_count'] == 0 else "⚠️",
                                    "✅" if results['clarity_details']['extra_count'] == 0 else "⚠️",
                                    "✅" if len(transcript.split()) >= len(paragraph['text'].split()) * 0.9 else "⚠️"
                                ]
                            }

                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)

                            st.markdown("---")

                            # ============================================================
                            # NEXT STEPS
                            # ============================================================
                            st.markdown("## 🎯 Next Steps")

                            next_col1, next_col2, next_col3 = st.columns(3)

                            with next_col1:
                                st.markdown("""
                                **🔄 Practice More:**
                                - Click 'Get New Paragraph' 
                                - Try different topics
                                - Record multiple attempts
                                """)

                            with next_col2:
                                st.markdown("""
                                **🎯 Focus Areas:**
                                - Work on weakest metric
                                - Reduce filler words
                                - Maintain steady pace
                                """)

                            with next_col3:
                                st.markdown("""
                                **📈 Track Progress:**
                                - Compare scores over time
                                - Set improvement goals
                                - Practice daily
                                """)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("Please check your API key and try again.")


if __name__ == "__main__":
    main()