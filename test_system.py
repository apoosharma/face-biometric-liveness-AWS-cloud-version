"""
================================================================================
TEST SUITE: test_system.py
Unit & Integration Verification for Face Biometric Auth & Liveness PAD System
================================================================================
"""

import os
import sys
import gc
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.database import (
    get_db_connection, init_db, log_attempt, get_all_logs,
    get_filtered_logs, register_user_db, get_all_users,
    delete_user_db, get_analytics_summary, clear_logs
)
from utils.liveness import (
    calculate_ear, LivenessDetector, DEFAULT_EAR_THRESHOLD
)
from utils.face_utils import (
    detect_faces, extract_face_embedding, match_face,
    aggregate_embeddings, load_embeddings, save_embeddings,
    delete_user_embedding, MATCH_DISTANCE_THRESHOLD
)


class TestBiometricDatabase(unittest.TestCase):
    """Verifies SQLite schema creation, data insertion, queries, and analytics."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.temp_dir.name, "test_biometric.db")
        init_db(self.test_db_path)

    def tearDown(self):
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_init_and_tables(self):
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row['name'] for row in cursor.fetchall()]
            self.assertIn("users", tables)
            self.assertIn("login_logs", tables)

    def test_user_crud(self):
        # 1. Register User
        res = register_user_db(
            user_id="USR-TEST-1",
            name="Jane Doe",
            sample_count=20,
            photo_path="/tmp/avatar.jpg",
            db_path=self.test_db_path
        )
        self.assertTrue(res)

        # 2. Get Users
        users = get_all_users(self.test_db_path)
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]['name'], "Jane Doe")
        self.assertEqual(users[0]['user_id'], "USR-TEST-1")

        # 3. Delete User
        del_res = delete_user_db("USR-TEST-1", self.test_db_path)
        self.assertTrue(del_res)
        users_after = get_all_users(self.test_db_path)
        self.assertEqual(len(users_after), 0)

    def test_log_attempt_and_analytics(self):
        # Log genuine attempt (Success)
        log_id1 = log_attempt(
            user_id="USR-01",
            matched_user="John Matrix",
            liveness_result="Real",
            confidence=0.96,
            status="Success",
            details="EAR: 0.18 blink confirmed",
            db_path=self.test_db_path
        )
        self.assertGreater(log_id1, 0)

        # Log spoof attempt
        log_id2 = log_attempt(
            user_id="USR-01",
            matched_user="John Matrix",
            liveness_result="Spoof",
            confidence=0.94,
            status="Spoof Alert",
            details="Static photo detected",
            db_path=self.test_db_path
        )
        self.assertGreater(log_id2, 0)

        # Check logs retrieval
        df_logs = get_all_logs(limit=10, db_path=self.test_db_path)
        self.assertEqual(len(df_logs), 2)

        # Check filtered logs
        df_spoofs = get_filtered_logs(status_filter="Spoof Alert", db_path=self.test_db_path)
        self.assertEqual(len(df_spoofs), 1)
        self.assertEqual(df_spoofs.iloc[0]['liveness_result'], "Spoof")

        # Check analytics summary (Average confidence of Successful attempts)
        summary = get_analytics_summary(self.test_db_path)
        self.assertEqual(summary['total_attempts'], 2)
        self.assertEqual(summary['successful_logins'], 1)
        self.assertEqual(summary['spoof_attempts'], 1)
        self.assertAlmostEqual(summary['avg_confidence'], 0.96, places=2)


class TestLivenessDetection(unittest.TestCase):
    """Verifies Soukupová & Čech EAR formula and Liveness state machine."""

    def test_ear_calculation_open_eye(self):
        # Synthetic landmark coordinates for an open eye (width = 20, height = 8)
        # p1=(0, 10), p2=(5, 14), p3=(15, 14), p4=(20, 10), p5=(15, 6), p6=(5, 6)
        # vertical distances = 8, horizontal = 20
        # EAR = (8 + 8) / (2 * 20) = 16 / 40 = 0.40
        open_eye_points = np.array([
            [0.0, 10.0],
            [5.0, 14.0],
            [15.0, 14.0],
            [20.0, 10.0],
            [15.0, 6.0],
            [5.0, 6.0]
        ])
        ear = calculate_ear(open_eye_points)
        self.assertAlmostEqual(ear, 0.40, places=2)
        self.assertGreater(ear, DEFAULT_EAR_THRESHOLD)

    def test_ear_calculation_closed_eye(self):
        # Synthetic landmark coordinates for closed eye (vertical distance near 0)
        closed_eye_points = np.array([
            [0.0, 10.0],
            [5.0, 10.2],
            [15.0, 10.2],
            [20.0, 10.0],
            [15.0, 9.8],
            [5.0, 9.8]
        ])
        ear = calculate_ear(closed_eye_points)
        self.assertLess(ear, DEFAULT_EAR_THRESHOLD)
        self.assertLess(ear, 0.10)

    def test_liveness_state_machine_reset(self):
        detector = LivenessDetector(ear_threshold=0.23, consec_frames=2, spoof_timeout=2.0)
        self.assertEqual(detector.blink_count, 0)
        self.assertFalse(detector.is_live)
        self.assertFalse(detector.is_spoof)

        detector.reset()
        self.assertEqual(detector.blink_count, 0)


class TestFaceUtils(unittest.TestCase):
    """Verifies Face Embeddings, Aggregation, and Metric Distance Matching."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_emb_path = os.path.join(self.temp_dir.name, "test_embeddings.pkl")

    def tearDown(self):
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_centroid_aggregation(self):
        # Generate 15 simulated noisy 128-d embeddings centered around a true vector
        rng = np.random.RandomState(42)
        true_vector = rng.randn(128)
        true_vector = true_vector / np.linalg.norm(true_vector)

        sample_embeddings = [
            true_vector + rng.normal(0, 0.02, 128) for _ in range(15)
        ]

        master = aggregate_embeddings(sample_embeddings)
        self.assertEqual(master.shape, (128,))
        # Master should be normalized to unit length
        self.assertAlmostEqual(np.linalg.norm(master), 1.0, places=4)

        # Cosine similarity between master and true_vector should be extremely high (> 0.99)
        cos_sim = np.dot(master, true_vector)
        self.assertGreater(cos_sim, 0.99)

    def test_embeddings_serialization(self):
        data = {
            "USR-101": {
                "name": "Sarah Connor",
                "embedding": np.ones(128) / np.sqrt(128),
                "enrolled_at": "2026-09-22 10:00:00",
                "sample_count": 20
            }
        }
        res = save_embeddings(data, self.test_emb_path)
        self.assertTrue(res)

        loaded = load_embeddings(self.test_emb_path)
        self.assertIn("USR-101", loaded)
        self.assertEqual(loaded["USR-101"]["name"], "Sarah Connor")

        del_res = delete_user_embedding("USR-101", self.test_emb_path)
        self.assertTrue(del_res)
        loaded_after = load_embeddings(self.test_emb_path)
        self.assertNotIn("USR-101", loaded_after)

    def test_face_matching(self):
        rng = np.random.RandomState(123)
        user_a_emb = rng.randn(128)
        user_a_emb /= np.linalg.norm(user_a_emb)

        user_b_emb = rng.randn(128)
        user_b_emb /= np.linalg.norm(user_b_emb)

        enrolled = {
            "USR-A": {"name": "Agent A", "embedding": user_a_emb},
            "USR-B": {"name": "Agent B", "embedding": user_b_emb}
        }

        # Query with slightly perturbed version of User A (distance ~ 0.22 < threshold 0.55)
        query_a = user_a_emb + rng.normal(0, 0.02, 128)
        query_a /= np.linalg.norm(query_a)

        uid, uname, conf, is_match = match_face(query_a, enrolled)
        self.assertTrue(is_match)
        self.assertEqual(uid, "USR-A")
        self.assertEqual(uname, "Agent A")
        self.assertGreater(conf, 0.65)

        # Query with totally different random vector
        query_unknown = rng.randn(128)
        query_unknown /= np.linalg.norm(query_unknown)
        # Ensure orthogonal
        query_unknown = query_unknown - np.dot(query_unknown, user_a_emb) * user_a_emb
        query_unknown /= np.linalg.norm(query_unknown)

        uid_u, uname_u, conf_u, is_match_u = match_face(query_unknown, enrolled)
        self.assertFalse(is_match_u)
        self.assertEqual(uid_u, "Unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
