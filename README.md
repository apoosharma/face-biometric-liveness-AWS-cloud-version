# face-biometric-liveness-AWS-cloud-version
AWS-ready biometric authentication system using face recognition and liveness detection, with a Streamlit dashboard for secure access control, attendance, audit logging, and presentation-attack monitoring. Designed for cloud integration with AWS services for scalable storage, deployment, monitoring, and security.

# 🛡️ BioGuard – Face Biometric Authentication with Liveness Detection

## 📌 Overview

**BioGuard** is a face-based biometric authentication and attendance system designed to provide secure, contactless identity verification using a standard laptop webcam.

The system combines:

- 👤 Face detection and recognition
- 🧠 Face embeddings for identity verification
- 👁️ Liveness detection using blink/Eye Aspect Ratio (EAR)
- 🛡️ Presentation Attack Detection (PAD)
- 📷 Real-time webcam authentication
- 🗃️ SQLite-based audit logging
- 📊 Security and authentication analytics
- 🌐 Streamlit-based interactive dashboard

The project is designed with an **AWS-ready cloud architecture**, allowing the local prototype to be extended into a scalable cloud-based biometric authentication platform.

---

# 🎯 Objectives

The main objectives of BioGuard are:

1. Develop a contactless biometric authentication system using a webcam.
2. Recognize registered users using facial features.
3. Detect whether the presented face belongs to a live person.
4. Prevent simple presentation attacks such as showing a photograph or static screen.
5. Maintain authentication and security audit logs.
6. Provide an interactive dashboard for monitoring authentication activity.
7. Design the system so that its storage, deployment, monitoring, and security components can be integrated with AWS cloud services.

---

# ✨ Key Features

## 👤 Face Recognition

The system captures a face using the webcam and compares its facial representation with enrolled users.

The authentication pipeline is:

```text
Webcam
   ↓
Face Detection
   ↓
Face Encoding / Embedding
   ↓
Identity Matching
   ↓
Liveness Verification
   ↓
Authentication Decision
