"""
================================================================================
BIOMETRIC FACE AUTHENTICATION & LIVENESS DETECTION SYSTEM
Main Streamlit Application: app.py
================================================================================
"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add current directory to path so utils can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from utils.database import (
    init_db, log_attempt, get_all_logs, get_filtered_logs,
    register_user_db, get_all_users, delete_user_db,
    get_analytics_summary, clear_logs, seed_mock_logs_if_empty
)
from utils.liveness import (
    LivenessDetector, draw_liveness_hud, DEFAULT_EAR_THRESHOLD
)
from utils.face_utils import (
    detect_faces, extract_face_embedding, match_face,
    aggregate_embeddings, load_embeddings, save_embeddings,
    delete_user_embedding, MATCH_DISTANCE_THRESHOLD
)

# Page configuration
st.set_page_config(
    page_title="BioGuard | Biometric Face Auth & Liveness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and mock data if clean slate
init_db()
seed_mock_logs_if_empty()

# Create snapshot directory if missing
SNAPSHOT_DIR = os.path.join(CURRENT_DIR, "data", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Custom Modern CSS Styling
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Glassmorphism Card Containers */
    .bio-card {
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    .bio-card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Status Badges */
    .badge-real {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-spoof {
        background-color: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 1px solid #da3633;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .badge-warning {
        background-color: rgba(210, 153, 34, 0.2);
        color: #d29922;
        border: 1px solid #bb8009;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Metrics Hero Banner */
    .metric-hero {
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.15), rgba(163, 113, 247, 0.15));
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        margin-top: 4px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <h2 style='margin: 0; color: #58a6ff;'>🛡️ BioGuard</h2>
            <p style='color: #8b949e; font-size: 0.85rem; margin-top: 4px;'>
                Biometric Auth & Liveness PAD
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation Menu",
        [
            "📸 Live Feed & Auth",
            "👤 User Enrollment",
            "📋 Audit Logs & History",
            "📊 Security Analytics"
        ],
        index=0
    )

    st.divider()
    
    # System Status in Sidebar
    users = get_all_users()
    embeddings = load_embeddings()
    st.markdown(f"""
        <div style='background: rgba(30,35,45,0.6); padding: 12px; border-radius: 8px; border: 1px solid #30363d;'>
            <div style='font-size: 0.85rem; color: #8b949e;'>SYSTEM STATUS</div>
            <div style='font-size: 0.95rem; margin-top: 4px;'>👥 Enrolled Users: <b>{len(users)}</b></div>
            <div style='font-size: 0.95rem;'>💾 Embedding Vectors: <b>{len(embeddings)}</b></div>
            <div style='font-size: 0.95rem;'>🔒 PAD Engine: <span style='color: #3fb950;'>Active (EAR)</span></div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: LIVE FEED & AUTHENTICATION
# ==============================================================================
if page == "📸 Live Feed & Auth":
    st.title("📸 Live Feed & Biometric Authentication")
    st.caption("Real-time facial recognition with Soukupová & Čech (2016) Eye Aspect Ratio (EAR) blink liveness detection.")

    col_video, col_telemetry = st.columns([1.6, 1.0])

    with col_telemetry:
        st.markdown("""
            <div class='bio-card'>
                <div class='bio-card-header'>⚙️ Authentication Controls</div>
            </div>
        """, unsafe_allow_html=True)
        
        feed_mode = st.radio(
            "Video Input Source:",
            ["🔴 Live Webcam Stream", "🖼️ Single Snapshot / File Upload", "🧪 Simulation / Test Mode"],
            help="Choose live camera feed or test sample image."
        )

        auto_auth = st.checkbox("⚡ Auto-Authenticate on Confirmed Blink", value=True,
                                help="Automatically logs successful attempt when a real blink is detected.")
        
        st.markdown("---")
        st.markdown("##### 🎯 Live Telemetry")
        telemetry_placeholder = st.empty()
        status_alert_placeholder = st.empty()

    with col_video:
        st.markdown("""
            <div class='bio-card'>
                <div class='bio-card-header'>📹 Live Video HUD</div>
            </div>
        """, unsafe_allow_html=True)
        video_placeholder = st.empty()

    # --- Mode A: Live Webcam Stream ---
    if feed_mode == "🔴 Live Webcam Stream":
        run_camera = st.toggle("🎥 Start Camera Stream", value=False)
        cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)

        if run_camera:
            cap = cv2.VideoCapture(int(cam_index))
            detector = LivenessDetector(ear_threshold=DEFAULT_EAR_THRESHOLD, consec_frames=2, spoof_timeout=3.5)
            enrolled_embeddings = load_embeddings()

            has_logged_success = False
            has_logged_spoof = False

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Unable to access webcam. Please verify camera permissions or switch to 'Single Snapshot' or 'Simulation Mode'.")
                    break

                # Flip for mirror selfie view
                frame = cv2.flip(frame, 1)

                # Process Liveness
                liveness_res = detector.process_frame(frame)
                
                # Match face if detected
                user_label = "Scanning..."
                conf_score = 0.0
                user_id = "Unknown"
                status_str = "Scanning"

                if liveness_res.get("face_box"):
                    box = liveness_res["face_box"]
                    emb = extract_face_embedding(frame, box)
                    if emb is not None:
                        user_id, user_label, conf_score, is_match = match_face(emb, enrolled_embeddings)
                        if not is_match:
                            user_label = "Unregistered / Unknown"

                # Render HUD
                hud_frame = draw_liveness_hud(frame, liveness_res, user_label, conf_score)

                # Convert to RGB for Streamlit display
                rgb_hud = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_hud, channels="RGB", use_container_width=True)

                # Update Telemetry Display
                is_live = liveness_res.get("is_live", False)
                is_spoof = liveness_res.get("is_spoof", False)
                current_ear = liveness_res.get("ear", 0.0)
                blinks = liveness_res.get("blink_count", 0)

                with telemetry_placeholder.container():
                    tcol1, tcol2 = st.columns(2)
                    tcol1.metric("Current EAR", f"{current_ear:.3f}")
                    tcol2.metric("Verified Blinks", f"{blinks}")
                    
                    if is_live:
                        st.markdown("<div class='badge-real'>🟢 LIVENESS VERIFIED: REAL HUMAN FACE</div>", unsafe_allow_html=True)
                    elif is_spoof:
                        st.markdown("<div class='badge-spoof'>🔴 SPOOF ALERT: STATIC PHOTO DETECTED</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='badge-warning'>🟡 AWAITING NATURAL BLINK...</div>", unsafe_allow_html=True)

                    st.markdown(f"**Identified User:** `{user_label}`")
                    st.markdown(f"**Confidence:** `{conf_score * 100:.1f}%`")

                # Handle Auto-Authentication Logic
                if auto_auth:
                    if is_live and not has_logged_success and user_id != "Unknown":
                        log_attempt(
                            user_id=user_id,
                            matched_user=user_label,
                            liveness_result="Real",
                            confidence=conf_score,
                            status="Success",
                            details=f"Auto-verified on blink (EAR: {current_ear:.2f})"
                        )
                        has_logged_success = True
                        status_alert_placeholder.success(f"✅ Access Granted: Welcome {user_label}!")

                    elif is_spoof and not has_logged_spoof:
                        log_attempt(
                            user_id=user_id,
                            matched_user=user_label,
                            liveness_result="Spoof",
                            confidence=conf_score,
                            status="Spoof Alert",
                            details="Static photo presentation attack intercepted (No blink in timeout window)"
                        )
                        has_logged_spoof = True
                        status_alert_placeholder.error("🚨 ACCESS DENIED: Presentation attack detected!")

                time.sleep(0.03)

            cap.release()
        else:
            video_placeholder.info("Click '🎥 Start Camera Stream' to activate live video authentication feed.")

    # --- Mode B: Single Snapshot / File Upload ---
    elif feed_mode == "🖼️ Single Snapshot / File Upload":
        uploaded_file = st.file_uploader("Upload an Image to Authenticate", type=["jpg", "jpeg", "png"])
        cam_snap = st.camera_input("Or take a single photo")
        
        target_img = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            target_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif cam_snap is not None:
            image = Image.open(cam_snap)
            target_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if target_img is not None:
            detector = LivenessDetector()
            enrolled_embeddings = load_embeddings()
            
            liveness_res = detector.process_frame(target_img)
            user_label = "Unknown"
            conf_score = 0.0
            user_id = "Unknown"
            is_match = False

            if liveness_res.get("face_box"):
                box = liveness_res["face_box"]
                emb = extract_face_embedding(target_img, box)
                if emb is not None:
                    user_id, user_label, conf_score, is_match = match_face(emb, enrolled_embeddings)

            hud_frame = draw_liveness_hud(target_img, liveness_res, user_label, conf_score)
            rgb_hud = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_hud, channels="RGB", use_container_width=True)

            with telemetry_placeholder.container():
                st.markdown(f"**Identified Subject:** `{user_label}`")
                st.markdown(f"**Confidence:** `{conf_score * 100:.1f}%`")
                st.info("ℹ️ Static images cannot demonstrate dynamic ocular blinks and require a live stream for EAR liveness verification.")

            if st.button("📝 Log This Verification Attempt", type="primary"):
                status_to_log = "Success" if is_match else "Failed"
                log_attempt(
                    user_id=user_id,
                    matched_user=user_label,
                    liveness_result="Real" if is_match else "Spoof",
                    confidence=conf_score,
                    status=status_to_log,
                    details="Snapshot manual authentication"
                )
                st.success(f"Recorded authentication log for {user_label}!")

    # --- Mode C: Simulation / Test Mode ---
    else:
        st.info("🧪 Test Simulation Mode: Allows testing the full pipeline, database logging, and analytics without needing an active webcam.")
        sim_user = st.selectbox("Select Enrolled User to Simulate", ["Dr. Robert Vance (USR-101)", "Sarah Connor (USR-102)", "Alex Mercer (USR-103)", "Unknown Attacker"])
        sim_liveness = st.radio("Simulated Liveness Result", ["🟢 Genuine Live Human (Passes EAR Blink)", "🔴 2D Printed Photo Spoof Attack (Fails EAR Blink)"])
        sim_confidence = st.slider("Simulated Match Confidence (%)", 30, 99, 94)

        if st.button("🚀 Execute Simulated Authentication Attempt", type="primary"):
            uid = sim_user.split("(")[-1].replace(")", "") if "(" in sim_user else "Unknown"
            uname = sim_user.split(" (")[0]
            is_real = "Genuine" in sim_liveness

            liveness_val = "Real" if is_real else "Spoof"
            status_val = "Success" if (is_real and uid != "Unknown") else ("Spoof Alert" if not is_real else "Failed")
            
            detail_msg = "Simulated EAR Blink 0.18" if is_real else "Simulated static photo (Zero blinks in 3.5s)"

            log_attempt(
                user_id=uid,
                matched_user=uname,
                liveness_result=liveness_val,
                confidence=sim_confidence / 100.0,
                status=status_val,
                details=detail_msg
            )
            
            if status_val == "Success":
                status_alert_placeholder.success(f"✅ [SIMULATED SUCCESS] Welcome, {uname}! Logged to SQLite database.")
            elif status_val == "Spoof Alert":
                status_alert_placeholder.error("🚨 [SIMULATED SPOOF ALERT] Presentation attack intercepted and logged!")
            else:
                status_alert_placeholder.warning("⚠️ [SIMULATED FAILURE] Unrecognized face attempt logged.")


# ==============================================================================
# PAGE 2: USER ENROLLMENT
# ==============================================================================
elif page == "👤 User Enrollment":
    st.title("👤 Biometric User Enrollment")
    st.caption("Register a new identity by capturing 15-20 frames to generate a normalized 128-d centroid embedding.")

    tab_enroll, tab_gallery = st.tabs(["📝 Register New Identity", "👥 Enrolled User Directory"])

    with tab_enroll:
        col_form, col_cam = st.columns([1.1, 1.4])

        with col_form:
            st.markdown("""
                <div class='bio-card'>
                    <div class='bio-card-header'>📋 Identity Metadata</div>
                </div>
            """, unsafe_allow_html=True)
            
            enroll_name = st.text_input("Full Name *", placeholder="e.g. Elena Rostova")
            enroll_id = st.text_input("User ID / Employee ID *", placeholder="e.g. EMP-2045")
            enroll_dept = st.selectbox("Department / Role", ["Engineering", "Security & Operations", "Executive", "Research & Dev", "Guest / Contractor"])
            frames_to_capture = st.slider("Sample Frames to Capture", min_value=10, max_value=30, value=20, step=5,
                                          help="Higher samples yield a more robust average centroid embedding.")

        with col_cam:
            st.markdown("""
                <div class='bio-card'>
                    <div class='bio-card-header'>📷 Multi-Frame Capture Feed</div>
                </div>
            """, unsafe_allow_html=True)

            enroll_cam_idx = st.number_input("Camera Index for Enrollment", min_value=0, max_value=5, value=0, step=1)
            btn_start_enroll = st.button("🚀 Start Multi-Frame Capture & Enroll", type="primary", use_container_width=True)
            enroll_preview = st.empty()
            enroll_progress = st.empty()
            enroll_status = st.empty()

        if btn_start_enroll:
            if not enroll_name.strip() or not enroll_id.strip():
                st.error("❌ Please provide both Full Name and User ID.")
            else:
                cap = cv2.VideoCapture(int(enroll_cam_idx))
                if not cap.isOpened():
                    st.error("⚠️ Could not open webcam for enrollment.")
                else:
                    captured_embeddings = []
                    captured_frames = []
                    enroll_status.info("📸 Please look directly at the camera with neutral expressions...")
                    
                    frame_count = 0
                    p_bar = enroll_progress.progress(0, text="Initializing Camera...")

                    while frame_count < frames_to_capture:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.flip(frame, 1)
                        boxes = detect_faces(frame)

                        if boxes:
                            box = boxes[0]
                            emb = extract_face_embedding(frame, box)
                            if emb is not None:
                                captured_embeddings.append(emb)
                                captured_frames.append(frame.copy())
                                frame_count += 1
                                
                                # Draw box on preview
                                x, y, bw, bh = box
                                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                                cv2.putText(frame, f"Capturing: {frame_count}/{frames_to_capture}", (x, max(0, y - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        rgb_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        enroll_preview.image(rgb_disp, channels="RGB", use_container_width=True)
                        
                        percent = int((frame_count / frames_to_capture) * 100)
                        p_bar.progress(percent, text=f"Capturing Frame {frame_count}/{frames_to_capture}...")
                        time.sleep(0.08)

                    cap.release()

                    if len(captured_embeddings) >= 5:
                        # Aggregate multi-frame centroid template
                        master_embedding = aggregate_embeddings(captured_embeddings)
                        
                        # Save avatar snapshot
                        snapshot_filename = f"{enroll_id.strip()}.jpg"
                        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_filename)
                        if captured_frames:
                            cv2.imwrite(snapshot_path, captured_frames[len(captured_frames)//2])

                        # Save to pickle embeddings dictionary
                        all_embeddings = load_embeddings()
                        all_embeddings[enroll_id.strip()] = {
                            "name": enroll_name.strip(),
                            "embedding": master_embedding,
                            "enrolled_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "sample_count": len(captured_embeddings)
                        }
                        save_embeddings(all_embeddings)

                        # Save to SQLite DB
                        register_user_db(
                            user_id=enroll_id.strip(),
                            name=enroll_name.strip(),
                            sample_count=len(captured_embeddings),
                            photo_path=snapshot_path
                        )

                        p_bar.progress(100, text="Enrollment Complete!")
                        enroll_status.success(f"🎉 Successfully enrolled **{enroll_name}** ({enroll_id}) with {len(captured_embeddings)} frames!")
                    else:
                        enroll_status.error("❌ Failed to detect a clear face across sufficient frames. Please try again in good lighting.")

    with tab_gallery:
        st.subheader("👥 Registered User Database")
        all_users = get_all_users()
        
        if not all_users:
            st.info("No users currently registered. Use the 'Register New Identity' tab to enroll.")
        else:
            cols = st.columns(3)
            for idx, u in enumerate(all_users):
                with cols[idx % 3]:
                    st.markdown(f"""
                        <div class='bio-card'>
                            <div style='font-size: 1.1rem; font-weight: 700; color: #58a6ff;'>{u['name']}</div>
                            <div style='font-size: 0.85rem; color: #8b949e; margin-bottom: 8px;'>ID: <code>{u['user_id']}</code></div>
                            <div style='font-size: 0.85rem;'>📅 Enrolled: {u['enrolled_at']}</div>
                            <div style='font-size: 0.85rem;'>🎞️ Sample Frames: {u['sample_count']}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    if u.get("photo_path") and os.path.exists(u["photo_path"]):
                        st.image(u["photo_path"], width=180)
                    
                    if st.button(f"🗑️ Delete User {u['user_id']}", key=f"del_{u['user_id']}"):
                        delete_user_db(u["user_id"])
                        delete_user_embedding(u["user_id"])
                        st.toast(f"Deleted user {u['name']}")
                        st.rerun()


# ==============================================================================
# PAGE 3: AUDIT LOGS & HISTORY
# ==============================================================================
elif page == "📋 Audit Logs & History":
    st.title("📋 Authentication Audit Logs & Security History")
    st.caption("Immutable local SQLite audit trail recording all access verification, liveness tests, and spoof alerts.")

    # Top KPI Cards
    summary = get_analytics_summary()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
            <div class='metric-hero'>
                <div class='metric-title'>Total Login Attempts</div>
                <div class='metric-value'>{summary['total_attempts']}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
            <div class='metric-hero'>
                <div class='metric-title'>Successful Logins</div>
                <div class='metric-value' style='color: #3fb950;'>{summary['successful_logins']}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
            <div class='metric-hero'>
                <div class='metric-title'>Spoof Attacks Blocked</div>
                <div class='metric-value' style='color: #f85149;'>{summary['spoof_attempts']}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
            <div class='metric-hero'>
                <div class='metric-title'>Avg Confidence</div>
                <div class='metric-value'>{summary['avg_confidence'] * 100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Filters Section
    with st.expander("🔍 Filter & Search Audit Trail", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1, 1, 1])
        with fcol1:
            search_query = st.text_input("Search (User Name, ID, Details)", placeholder="e.g. Alice or EMP-101")
        with fcol2:
            status_filter = st.selectbox("Status", ["All", "Success", "Spoof Alert", "Failed", "Unrecognized"])
        with fcol3:
            liveness_filter = st.selectbox("Liveness PAD", ["All", "Real", "Spoof"])
        with fcol4:
            limit_val = st.number_input("Max Rows", min_value=20, max_value=2000, value=200, step=50)

    # Fetch Filtered Logs
    df_logs = get_filtered_logs(
        search_query=search_query,
        status_filter=status_filter,
        liveness_filter=liveness_filter,
        limit=int(limit_val)
    )

    if df_logs.empty:
        st.info("No audit logs match the selected filter criteria.")
    else:
        # Format columns for crisp display
        df_display = df_logs.copy()
        df_display['confidence_pct'] = df_display['confidence'].apply(lambda c: f"{c * 100:.1f}%")
        
        st.dataframe(
            df_display[['timestamp', 'user_id', 'matched_user', 'liveness_result', 'confidence_pct', 'status', 'details']],
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp", help="Time of authentication event"),
                "user_id": st.column_config.TextColumn("User ID"),
                "matched_user": st.column_config.TextColumn("Matched Identity"),
                "liveness_result": st.column_config.TextColumn("Liveness (PAD)"),
                "confidence_pct": st.column_config.TextColumn("Confidence"),
                "status": st.column_config.TextColumn("Action Status"),
                "details": st.column_config.TextColumn("Diagnostic Details")
            },
            hide_index=True
        )

        # Export and Clear utilities
        bcol1, bcol2, _ = st.columns([1.2, 1.2, 3])
        with bcol1:
            csv_data = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Logs to CSV",
                data=csv_data,
                file_name=f"biometric_audit_logs_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with bcol2:
            if st.button("🗑️ Clear All Logs", type="secondary", use_container_width=True):
                clear_logs()
                st.toast("Cleared all audit records.")
                st.rerun()


# ==============================================================================
# PAGE 4: SECURITY & SYSTEM ANALYTICS
# ==============================================================================
elif page == "📊 Security Analytics":
    st.title("📊 Biometric Security & Performance Analytics")
    st.caption("Statistical telemetry, temporal trends, and Presentation Attack Detection (PAD) metrics.")

    df_logs = get_all_logs(limit=1000)

    if df_logs.empty:
        st.info("No log data available to generate analytics. Log in via 'Live Feed' or 'Simulation Mode' to populate data.")
    else:
        df_logs['datetime'] = pd.to_datetime(df_logs['timestamp'])
        df_logs['date'] = df_logs['datetime'].dt.date
        df_logs['hour'] = df_logs['datetime'].dt.hour

        # Row 1: Logins Over Time (Success vs Failed vs Spoof)
        col_time, col_pie = st.columns([1.8, 1.2])

        with col_time:
            st.markdown("### 📈 Authentication Events Over Time")
            # Aggregate by date and status
            time_agg = df_logs.groupby(['date', 'status']).size().reset_index(name='count')
            time_agg['date'] = time_agg['date'].astype(str)

            color_map = {
                "Success": "#3fb950",
                "Spoof Alert": "#f85149",
                "Failed": "#d29922",
                "Unrecognized": "#a371f7"
            }

            fig_time = px.bar(
                time_agg,
                x='date',
                y='count',
                color='status',
                color_discrete_map=color_map,
                barmode='group',
                title="Daily Login Attempts by Outcome",
                labels={"count": "Number of Attempts", "date": "Date", "status": "Event Status"},
                template="plotly_dark"
            )
            fig_time.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_time, use_container_width=True)

        with col_pie:
            st.markdown("### 🛡️ Liveness Outcome Ratio")
            liveness_counts = df_logs['liveness_result'].value_counts().reset_index()
            liveness_counts.columns = ['Result', 'Count']

            fig_pie = px.pie(
                liveness_counts,
                names='Result',
                values='Count',
                color='Result',
                color_discrete_map={"Real": "#3fb950", "Spoof": "#f85149"},
                hole=0.45,
                title="Genuine (Live) vs Spoof Presentation Attacks",
                template="plotly_dark"
            )
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # Row 2: Per-User Frequency & Confidence Distribution
        col_user, col_conf = st.columns([1.5, 1.5])

        with col_user:
            st.markdown("### 👥 Per-User Login Frequency")
            user_counts = df_logs[df_logs['matched_user'] != 'Unknown']['matched_user'].value_counts().reset_index()
            user_counts.columns = ['User', 'Logins']

            if not user_counts.empty:
                fig_user = px.bar(
                    user_counts,
                    x='Logins',
                    y='User',
                    orientation='h',
                    color='Logins',
                    color_continuous_scale="Blues",
                    title="Top Enrolled Users Activity Frequency",
                    template="plotly_dark"
                )
                fig_user.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_user, use_container_width=True)
            else:
                st.info("No registered user logins logged yet.")

        with col_conf:
            st.markdown("### 🎯 Model Confidence Score Distribution")
            fig_conf = px.histogram(
                df_logs,
                x='confidence',
                color='status',
                color_discrete_map=color_map,
                nbins=20,
                marginal="box",
                title="Confidence Spread Across Authentication Outcomes",
                labels={"confidence": "Confidence Score (0.0 - 1.0)"},
                template="plotly_dark"
            )
            fig_conf.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_conf, use_container_width=True)


