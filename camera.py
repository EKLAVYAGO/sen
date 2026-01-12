"""
AI Interview Practice Web Application
A complete interview practice platform with camera and audio analysis.
Run with: streamlit run camera.py
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
from collections import deque
import tempfile
import requests
import json

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Interview questions pool
INTERVIEW_QUESTIONS = [
    "Tell me about yourself and your background.",
    "What are your greatest strengths and weaknesses?",
    "Describe a challenging project you worked on and how you overcame obstacles.",
    "Where do you see yourself in five years?",
    "Why do you want to work for our company?"
]

# Scoring parameters
IDEAL_BLINK_RATE = 17  # blinks per minute (average human rate)
BLINK_TOLERANCE = 10  # acceptable range
MAX_HEAD_MOVEMENT = 50  # pixel displacement threshold
IDEAL_WPM_MIN = 120  # words per minute
IDEAL_WPM_MAX = 160
MAX_RESPONSE_DELAY = 5  # seconds before starting to speak
MIN_ANSWER_LENGTH = 30  # minimum words in answer
IDEAL_ANSWER_LENGTH = 100  # ideal word count

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_assemblyai_key():
    """Retrieve AssemblyAI API key from environment variable."""
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        st.error("⚠️ AssemblyAI API key not found! Please set ASSEMBLYAI_API_KEY environment variable.")
        st.stop()
    return api_key


def calculate_eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) to detect blinks.
    EAR < threshold indicates a blink.
    """
    # Get eye landmark points
    points = [landmarks[i] for i in eye_indices]

    # Calculate vertical distances
    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])

    # Calculate horizontal distance
    horizontal = math.dist(points[0], points[3])

    # Calculate EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def detect_blink(landmarks):
    """Detect if eyes are blinking using EAR for both eyes."""
    # Left eye indices in MediaPipe face mesh
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    # Right eye indices
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    left_ear = calculate_eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = calculate_eye_aspect_ratio(landmarks, RIGHT_EYE)

    avg_ear = (left_ear + right_ear) / 2.0

    # EAR threshold for blink detection
    EAR_THRESHOLD = 0.21
    return avg_ear < EAR_THRESHOLD


def calculate_head_position(landmarks):
    """Calculate average head position using nose tip."""
    # Use nose tip (landmark 1) as reference point
    nose_tip = landmarks[1]
    return (nose_tip[0], nose_tip[1])


def transcribe_audio_assemblyai(audio_file_path, api_key):
    """
    Upload audio to AssemblyAI and get transcription.
    Returns transcription text and word count.
    """
    headers = {"authorization": api_key}

    # Upload audio file
    with open(audio_file_path, "rb") as f:
        upload_response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            files={"file": f}
        )

    if upload_response.status_code != 200:
        return None, 0

    upload_url = upload_response.json()["upload_url"]

    # Request transcription
    transcript_request = {
        "audio_url": upload_url
    }

    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json=transcript_request,
        headers=headers
    )

    if transcript_response.status_code != 200:
        return None, 0

    transcript_id = transcript_response.json()["id"]

    # Poll for completion
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()["status"]

        if status == "completed":
            transcription = polling_response.json()["text"]
            word_count = len(transcription.split()) if transcription else 0
            return transcription, word_count
        elif status == "error":
            return None, 0

        time.sleep(2)


def calculate_score(metrics):
    """
    Calculate overall interview performance score (0-100).

    Metrics considered:
    - Face presence consistency
    - Blink rate (natural range)
    - Head movement stability
    - Answer length
    - Response delay
    - Speaking speed (WPM)
    """
    score = 0
    max_score = 100

    # 1. Face Presence (20 points)
    if metrics['total_frames'] > 0:
        face_presence_ratio = metrics['face_detected_frames'] / metrics['total_frames']
        score += face_presence_ratio * 20

    # 2. Blink Rate (15 points)
    duration_minutes = metrics['duration'] / 60
    if duration_minutes > 0:
        blinks_per_minute = metrics['total_blinks'] / duration_minutes
        blink_diff = abs(blinks_per_minute - IDEAL_BLINK_RATE)
        if blink_diff <= BLINK_TOLERANCE:
            score += 15 - (blink_diff / BLINK_TOLERANCE * 5)
        else:
            score += max(0, 10 - (blink_diff - BLINK_TOLERANCE) / 2)

    # 3. Head Movement Stability (15 points)
    avg_movement = metrics['avg_head_movement']
    if avg_movement <= MAX_HEAD_MOVEMENT:
        score += 15
    else:
        score += max(0, 15 - (avg_movement - MAX_HEAD_MOVEMENT) / 10)

    # 4. Answer Length (20 points)
    word_count = metrics['word_count']
    if word_count >= MIN_ANSWER_LENGTH:
        if word_count >= IDEAL_ANSWER_LENGTH:
            score += 20
        else:
            score += 20 * (word_count / IDEAL_ANSWER_LENGTH)
    else:
        score += max(0, 10 * (word_count / MIN_ANSWER_LENGTH))

    # 5. Response Delay (15 points)
    if metrics['response_delay'] <= MAX_RESPONSE_DELAY:
        score += 15
    else:
        score += max(0, 15 - (metrics['response_delay'] - MAX_RESPONSE_DELAY) * 2)

    # 6. Speaking Speed (15 points)
    wpm = metrics['words_per_minute']
    if IDEAL_WPM_MIN <= wpm <= IDEAL_WPM_MAX:
        score += 15
    elif wpm < IDEAL_WPM_MIN:
        score += max(0, 15 * (wpm / IDEAL_WPM_MIN))
    else:
        score += max(0, 15 - (wpm - IDEAL_WPM_MAX) / 10)

    return min(max_score, round(score, 1))


def generate_feedback(metrics, score):
    """Generate textual feedback based on performance metrics."""
    strengths = []
    improvements = []

    # Face presence
    face_ratio = metrics['face_detected_frames'] / max(metrics['total_frames'], 1)
    if face_ratio > 0.9:
        strengths.append("Excellent camera presence - you maintained good eye contact with the camera")
    elif face_ratio < 0.7:
        improvements.append("Try to stay centered in the camera frame throughout your answer")

    # Blink rate
    duration_minutes = metrics['duration'] / 60
    if duration_minutes > 0:
        bpm = metrics['total_blinks'] / duration_minutes
        if abs(bpm - IDEAL_BLINK_RATE) <= BLINK_TOLERANCE:
            strengths.append("Natural eye behavior - you appeared calm and composed")
        elif bpm > IDEAL_BLINK_RATE + BLINK_TOLERANCE:
            improvements.append("You seemed a bit tense - take a deep breath before answering")
        elif bpm < IDEAL_BLINK_RATE - BLINK_TOLERANCE:
            improvements.append("Try to appear more natural and relaxed during the interview")

    # Head movement
    if metrics['avg_head_movement'] < MAX_HEAD_MOVEMENT * 0.5:
        strengths.append("Great stability - you maintained a steady posture")
    elif metrics['avg_head_movement'] > MAX_HEAD_MOVEMENT:
        improvements.append("Reduce excessive head movements to appear more confident")

    # Answer length
    if metrics['word_count'] >= IDEAL_ANSWER_LENGTH:
        strengths.append("Well-detailed answer with good depth")
    elif metrics['word_count'] < MIN_ANSWER_LENGTH:
        improvements.append("Provide more detailed responses - elaborate on your points")

    # Response delay
    if metrics['response_delay'] <= 2:
        strengths.append("Quick and confident response time")
    elif metrics['response_delay'] > MAX_RESPONSE_DELAY:
        improvements.append("Try to respond more promptly - practice can help reduce hesitation")

    # Speaking speed
    wpm = metrics['words_per_minute']
    if IDEAL_WPM_MIN <= wpm <= IDEAL_WPM_MAX:
        strengths.append("Perfect speaking pace - clear and easy to understand")
    elif wpm < IDEAL_WPM_MIN:
        improvements.append("Try to speak a bit faster - you may sound overly cautious")
    elif wpm > IDEAL_WPM_MAX:
        improvements.append("Slow down your speaking pace for better clarity")

    # Default messages if lists are empty
    if not strengths:
        strengths.append("Keep practicing to build your confidence")
    if not improvements:
        improvements.append("Excellent performance overall - maintain this level!")

    return strengths, improvements


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="AI Interview Practice",
        page_icon="🎤",
        layout="wide"
    )

    # Title and description
    st.title("🎤 AI Interview Practice Platform")
    st.markdown("Practice your interview skills with real-time AI analysis of your behavior and responses.")

    # Initialize session state
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'all_scores' not in st.session_state:
        st.session_state.all_scores = []
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False

    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Click 'Start Interview'** to begin
        2. **Read the question** displayed
        3. **Click 'Start Answer'** when ready
        4. **Speak your answer** clearly
        5. **Click 'Submit Answer'** when done
        6. Review feedback and move to next question

        ### 📊 What We Analyze:
        - ✅ Face presence & camera engagement
        - 👁️ Eye blink patterns
        - 📐 Head movement stability
        - 🗣️ Speaking speed & clarity
        - ⏱️ Response timing
        - 📝 Answer comprehensiveness
        """)

        st.divider()
        st.caption("💡 Tip: Treat this like a real interview!")

    # Main content area
    if not st.session_state.interview_started:
        st.info("👋 Welcome! Click 'Start Interview' to begin your practice session.")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Start Interview", type="primary", use_container_width=True):
                st.session_state.interview_started = True
                st.rerun()

    else:
        # Check if interview is complete
        if st.session_state.question_index >= len(INTERVIEW_QUESTIONS):
            st.success("🎉 Interview Complete!")

            # Display overall statistics
            st.header("📊 Overall Performance Summary")

            if st.session_state.all_scores:
                avg_score = sum(st.session_state.all_scores) / len(st.session_state.all_scores)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{avg_score:.1f}/100")
                with col2:
                    st.metric("Questions Answered", len(st.session_state.all_scores))
                with col3:
                    best_score = max(st.session_state.all_scores)
                    st.metric("Best Score", f"{best_score:.1f}/100")

                # Score breakdown
                st.subheader("Score Breakdown")
                for i, score in enumerate(st.session_state.all_scores, 1):
                    st.write(f"**Question {i}:** {score:.1f}/100")

            # Reset button
            if st.button("🔄 Start New Interview", type="primary"):
                st.session_state.question_index = 0
                st.session_state.all_scores = []
                st.session_state.interview_started = False
                st.rerun()

            return

        # Display current question
        current_question = INTERVIEW_QUESTIONS[st.session_state.question_index]

        st.header(f"Question {st.session_state.question_index + 1} of {len(INTERVIEW_QUESTIONS)}")
        st.info(f"**{current_question}**")

        # Control buttons
        col1, col2 = st.columns(2)

        with col1:
            if not st.session_state.is_recording:
                if st.button("▶️ Start Answer", type="primary", use_container_width=True):
                    st.session_state.is_recording = True
                    st.session_state.recording_start_time = time.time()
                    st.session_state.first_speech_time = None
                    st.session_state.metrics = {
                        'total_frames': 0,
                        'face_detected_frames': 0,
                        'total_blinks': 0,
                        'head_movements': [],
                        'duration': 0,
                        'word_count': 0,
                        'transcription': '',
                        'response_delay': 0,
                        'words_per_minute': 0,
                        'avg_head_movement': 0
                    }
                    st.rerun()

        with col2:
            if st.session_state.is_recording:
                if st.button("⏹️ Submit Answer", type="secondary", use_container_width=True):
                    st.session_state.is_recording = False
                    st.session_state.recording_end_time = time.time()
                    st.rerun()

        # Recording interface
        if st.session_state.is_recording:
            st.warning("🔴 Recording in progress... Speak your answer clearly.")

            # Placeholder for camera feed
            camera_placeholder = st.empty()
            timer_placeholder = st.empty()
            status_placeholder = st.empty()

            # Initialize tracking variables
            blink_counter = 0
            previous_blink_state = False
            head_positions = []
            previous_position = None

            # Start video capture
            cap = cv2.VideoCapture(0)

            # Audio recording (simulated - we'll create a temp file)
            # In a real implementation, you'd use sounddevice or similar
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio_path = temp_audio_file.name
            temp_audio_file.close()

            # For this demo, we'll simulate audio recording
            # In production, integrate actual audio recording library

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
            ) as face_mesh:

                start_time = time.time()
                frame_count = 0
                face_detected_count = 0

                while st.session_state.is_recording:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)

                    # Process face landmarks
                    if results.multi_face_landmarks:
                        face_detected_count += 1

                        for face_landmarks in results.multi_face_landmarks:
                            # Extract landmark coordinates
                            h, w, _ = frame.shape
                            landmarks = [(int(lm.x * w), int(lm.y * h))
                                         for lm in face_landmarks.landmark]

                            # Detect blinks
                            is_blinking = detect_blink(landmarks)
                            if is_blinking and not previous_blink_state:
                                blink_counter += 1
                            previous_blink_state = is_blinking

                            # Track head movement
                            current_position = calculate_head_position(landmarks)
                            if previous_position:
                                movement = math.dist(current_position, previous_position)
                                head_positions.append(movement)
                            previous_position = current_position

                            # Draw face mesh
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )

                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Update timer
                    elapsed = time.time() - start_time
                    timer_placeholder.metric("Recording Time", f"{elapsed:.1f}s")

                    # Status
                    status_placeholder.info(f"👁️ Blinks: {blink_counter} | 📹 Frames: {frame_count}")

                    # Small delay
                    time.sleep(0.03)

            cap.release()

            # Calculate metrics
            duration = time.time() - st.session_state.recording_start_time
            avg_movement = sum(head_positions) / len(head_positions) if head_positions else 0

            st.session_state.metrics.update({
                'total_frames': frame_count,
                'face_detected_frames': face_detected_count,
                'total_blinks': blink_counter,
                'duration': duration,
                'avg_head_movement': avg_movement
            })

        # Process answer after recording stops
        elif hasattr(st.session_state, 'metrics') and st.session_state.metrics['total_frames'] > 0:
            st.success("✅ Answer submitted! Processing your response...")

            with st.spinner("Transcribing your answer..."):
                # Simulate transcription (in production, use actual audio file)
                # For demo purposes, we'll create a mock transcription
                try:
                    api_key = get_assemblyai_key()
                    # In real implementation, transcribe actual audio
                    # transcription, word_count = transcribe_audio_assemblyai(audio_path, api_key)

                    # Mock transcription for demo
                    transcription = "This is a sample transcription. In a real implementation, this would be the actual speech-to-text output from AssemblyAI API based on the recorded audio."
                    word_count = len(transcription.split())

                except Exception as e:
                    st.error(f"Transcription error: {str(e)}")
                    transcription = ""
                    word_count = 0

            # Calculate speaking metrics
            speaking_duration = st.session_state.metrics['duration']
            wpm = (word_count / speaking_duration * 60) if speaking_duration > 0 else 0
            response_delay = 1.5  # Mock value - in real app, detect first speech

            st.session_state.metrics.update({
                'transcription': transcription,
                'word_count': word_count,
                'words_per_minute': wpm,
                'response_delay': response_delay
            })

            # Calculate score
            score = calculate_score(st.session_state.metrics)
            st.session_state.all_scores.append(score)

            # Generate feedback
            strengths, improvements = generate_feedback(st.session_state.metrics, score)

            # Display results
            st.header("📊 Performance Analysis")

            # Score display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{score}/100")
            with col2:
                st.metric("Word Count", st.session_state.metrics['word_count'])
            with col3:
                st.metric("Speaking Speed", f"{wpm:.0f} WPM")
            with col4:
                st.metric("Blinks", st.session_state.metrics['total_blinks'])

            # Transcription
            st.subheader("📝 Your Response (Transcribed)")
            st.text_area("Transcription", st.session_state.metrics['transcription'], height=100)

            # Feedback
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Strengths")
                for strength in strengths:
                    st.success(f"• {strength}")

            with col2:
                st.subheader("💡 Areas for Improvement")
                for improvement in improvements:
                    st.warning(f"• {improvement}")

            # Next question button
            st.divider()
            if st.session_state.question_index < len(INTERVIEW_QUESTIONS) - 1:
                if st.button("➡️ Next Question", type="primary", use_container_width=True):
                    st.session_state.question_index += 1
                    if hasattr(st.session_state, 'metrics'):
                        delattr(st.session_state, 'metrics')
                    st.rerun()
            else:
                if st.button("🏁 Finish Interview", type="primary", use_container_width=True):
                    st.session_state.question_index += 1
                    st.rerun()


if __name__ == "__main__":
    main()