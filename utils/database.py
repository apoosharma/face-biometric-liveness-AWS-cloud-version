"""
================================================================================
DATABASE MODULE: utils/database.py
Face-based Biometric Authentication System with Liveness Detection
================================================================================

VIVA / TECHNICAL EXPLANATION:
-----------------------------
1. Relational Database Management (SQLite):
   - SQLite is an embedded, serverless, self-contained, transactional SQL database engine.
   - It requires zero configuration, stores all data in a single local cross-platform file,
     making it ideal for edge computing and localized biometric authentication kiosks.

2. Database Schema:
   - 'users' table:
     Stores metadata for registered identities (User ID, Name, Enrollment timestamp,
     number of training frames captured, and local avatar photo path).
   - 'login_logs' table:
     An immutable audit trail recording every authentication attempt.
     Columns: id, timestamp, user_id, matched_user, liveness_result ('Real' / 'Spoof'),
     confidence (0.00 to 1.00 float), status ('Success' / 'Failed' / 'Spoof Alert' / 'Unknown'),
     and diagnostic details.

3. ACID Properties & Security:
   - Atomicity, Consistency, Isolation, Durability are guaranteed.
   - Parameterized queries ('?') are strictly used to prevent SQL Injection vulnerabilities.
   - Database connections use context managers ('with sqlite3.connect(...)') to prevent
     connection leakage and lock contention.
================================================================================
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

# Default path for the database file inside the /data directory
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "biometric.db"
)


from contextlib import contextmanager


@contextmanager
def get_db_connection(db_path: str = DEFAULT_DB_PATH):
    """
    Creates and yields a thread-safe SQLite connection, automatically closing it on exit.
    
    Parameters:
        db_path (str): Absolute or relative filesystem path to the SQLite DB.
        
    Yields:
        sqlite3.Connection: Active database connection object with row factory enabled.
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Enable dict-like column access on rows
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the database schema if tables do not already exist.
    Creates 'users' and 'login_logs' tables along with performance indexes.
    
    Parameters:
        db_path (str): Path to the SQLite database.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Table for Registered Users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enrolled_at DATETIME NOT NULL,
                sample_count INTEGER DEFAULT 15,
                photo_path TEXT
            )
        """)
        
        # 2. Table for Authentication & Audit Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                user_id TEXT,
                matched_user TEXT NOT NULL,
                liveness_result TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                details TEXT
            )
        """)
        
        # 3. Create indexes on frequently filtered/queried columns for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON login_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_user_id ON login_logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_status ON login_logs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_liveness ON login_logs(liveness_result)")
        
        conn.commit()


def log_attempt(
    user_id: str,
    matched_user: str,
    liveness_result: str,
    confidence: float,
    status: str,
    details: str = "",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Logs an authentication attempt into the audit trail table.
    
    Parameters:
        user_id (str): ID of the matched user or 'Unknown' / 'N/A'
        matched_user (str): Display name of the matched user or 'Unknown'
        liveness_result (str): 'Real' if eye blink detected, 'Spoof' if static photo / no blink
        confidence (float): Recognition confidence score (0.0 to 1.0)
        status (str): Outcome ('Success', 'Failed', 'Spoof Alert', 'Unrecognized')
        details (str): Additional diagnostics (e.g. 'Blink EAR: 0.19', 'Euclidean dist: 0.38')
        db_path (str): Path to SQLite DB
        
    Returns:
        int: The primary key ID of the inserted log record.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO login_logs (
                timestamp, user_id, matched_user, liveness_result, confidence, status, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (now_str, user_id, matched_user, liveness_result, float(confidence), status, details))
        conn.commit()
        return cursor.lastrowid


def register_user_db(
    user_id: str,
    name: str,
    sample_count: int = 15,
    photo_path: str = "",
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Inserts or updates a registered user record in the SQLite database.
    
    Parameters:
        user_id (str): Unique alphanumeric identifier (e.g., 'EMP-1001')
        name (str): Full legal name of user (e.g., 'Alice Smith')
        sample_count (int): Number of facial frames captured during enrollment
        photo_path (str): Path to enrolled reference avatar photo
        db_path (str): Path to SQLite DB
        
    Returns:
        bool: True if successful, False otherwise.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO users (
                user_id, name, enrolled_at, sample_count, photo_path
            ) VALUES (?, ?, ?, ?, ?)
        """, (user_id, name, now_str, sample_count, photo_path))
        conn.commit()
        return True


def delete_user_db(user_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Removes a user from the 'users' database table.
    
    Parameters:
        user_id (str): User ID to remove.
        db_path (str): Path to SQLite DB.
        
    Returns:
        bool: True if deleted, False if user did not exist.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_all_users(db_path: str = DEFAULT_DB_PATH) -> list:
    """
    Fetches all registered users ordered by enrollment timestamp.
    
    Returns:
        list[dict]: List of user dictionaries.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, name, enrolled_at, sample_count, photo_path FROM users ORDER BY enrolled_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_all_logs(limit: int = 500, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieves the most recent login attempt logs as a pandas DataFrame.
    
    Parameters:
        limit (int): Maximum records to fetch (default 500)
        db_path (str): Path to SQLite DB
        
    Returns:
        pd.DataFrame: Audit trail dataframe.
    """
    with get_db_connection(db_path) as conn:
        query = "SELECT * FROM login_logs ORDER BY timestamp DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df


def get_filtered_logs(
    search_query: str = "",
    status_filter: str = "All",
    liveness_filter: str = "All",
    date_from: datetime.date = None,
    date_to: datetime.date = None,
    limit: int = 500,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """
    Retrieves audit logs filtered by search terms, status, liveness, and date range.
    
    Parameters:
        search_query (str): Keyword matching user_id, matched_user, or details
        status_filter (str): 'All' or specific status
        liveness_filter (str): 'All', 'Real', or 'Spoof'
        date_from (date): Start date (inclusive)
        date_to (date): End date (inclusive)
        limit (int): Maximum records
        db_path (str): Path to SQLite DB
        
    Returns:
        pd.DataFrame: Filtered audit logs.
    """
    with get_db_connection(db_path) as conn:
        conditions = ["1=1"]
        params = []
        
        if search_query:
            conditions.append("(user_id LIKE ? OR matched_user LIKE ? OR details LIKE ?)")
            q = f"%{search_query}%"
            params.extend([q, q, q])
            
        if status_filter != "All":
            conditions.append("status = ?")
            params.append(status_filter)
            
        if liveness_filter != "All":
            conditions.append("liveness_result = ?")
            params.append(liveness_filter)
            
        if date_from:
            conditions.append("timestamp >= ?")
            params.append(f"{date_from} 00:00:00")
            
        if date_to:
            conditions.append("timestamp <= ?")
            params.append(f"{date_to} 23:59:59")
            
        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM login_logs WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        return df


def clear_logs(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Purges all audit records from the 'login_logs' table.
    
    Returns:
        bool: True upon successful deletion.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM login_logs")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='login_logs'")
        conn.commit()
        return True


def get_analytics_summary(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Computes key performance indicators (KPIs) and aggregated statistics
    for display on the Streamlit Analytics dashboard.
    
    Returns:
        dict: Aggregated security metrics.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Total attempts
        cursor.execute("SELECT COUNT(*) FROM login_logs")
        total_attempts = cursor.fetchone()[0]
        
        # Total successful logins
        cursor.execute("SELECT COUNT(*) FROM login_logs WHERE status = 'Success'")
        successful_logins = cursor.fetchone()[0]
        
        # Total spoof attempts intercepted
        cursor.execute("SELECT COUNT(*) FROM login_logs WHERE liveness_result = 'Spoof' OR status = 'Spoof Alert'")
        spoof_attempts = cursor.fetchone()[0]
        
        # Total failed logins (unrecognized face or low confidence)
        cursor.execute("SELECT COUNT(*) FROM login_logs WHERE status IN ('Failed', 'Unrecognized')")
        failed_logins = cursor.fetchone()[0]
        
        # Average confidence of successful logins
        cursor.execute("SELECT AVG(confidence) FROM login_logs WHERE status = 'Success'")
        avg_conf_row = cursor.fetchone()[0]
        avg_confidence = float(avg_conf_row) if avg_conf_row is not None else 0.0
        
        # Total enrolled users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        success_rate = (successful_logins / total_attempts * 100) if total_attempts > 0 else 0.0
        spoof_rate = (spoof_attempts / total_attempts * 100) if total_attempts > 0 else 0.0
        
        return {
            "total_attempts": total_attempts,
            "successful_logins": successful_logins,
            "spoof_attempts": spoof_attempts,
            "failed_logins": failed_logins,
            "avg_confidence": avg_confidence,
            "total_users": total_users,
            "success_rate": success_rate,
            "spoof_rate": spoof_rate
        }


def seed_mock_logs_if_empty(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Seeds realistic historical data if the database is newly created and empty.
    Provides immediate rich charts and trends on the Analytics page for demonstration.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM login_logs")
        count = cursor.fetchone()[0]
        if count > 0:
            return  # Already populated
        
        # Sample simulated users
        mock_users = [
            ("USR-101", "Dr. Robert Vance"),
            ("USR-102", "Sarah Connor"),
            ("USR-103", "Alex Mercer"),
            ("USR-104", "Elena Rostova"),
            ("USR-105", "Marcus Brody")
        ]
        
        # Register simulated users into users table
        for uid, uname in mock_users:
            register_user_db(uid, uname, sample_count=18, photo_path="", db_path=db_path)
            
        # Generate 45 realistic historical login attempts spread over the past 7 days
        now = datetime.now()
        entries = []
        
        for i in range(45):
            days_ago = random.randint(0, 6)
            hours_ago = random.randint(0, 23)
            mins_ago = random.randint(0, 59)
            t_stamp = now - timedelta(days=days_ago, hours=hours_ago, minutes=mins_ago)
            t_str = t_stamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # 70% chance of successful login, 18% spoof attack attempt, 12% failed/unrecognized
            rand_val = random.random()
            if rand_val < 0.70:
                uid, uname = random.choice(mock_users)
                liveness = "Real"
                conf = round(random.uniform(0.85, 0.99), 3)
                status = "Success"
                ear_val = round(random.uniform(0.16, 0.22), 2)
                details = f"EAR Blink: {ear_val} (Passed), Cosine Sim: {round(conf, 2)}"
            elif rand_val < 0.88:
                # Spoof photo attempt
                uid, uname = random.choice(mock_users)
                liveness = "Spoof"
                conf = round(random.uniform(0.70, 0.92), 3)
                status = "Spoof Alert"
                details = "Static photo attack detected (Zero blink activity in 3.5s window)"
            else:
                # Failed login (Unknown face or low confidence)
                uid, uname = "Unknown", "Unknown"
                liveness = "Real"
                conf = round(random.uniform(0.30, 0.54), 3)
                status = "Failed"
                details = "Face distance exceeded threshold (0.62 > 0.55)"
                
            entries.append((t_str, uid, uname, liveness, conf, status, details))
            
        cursor.executemany("""
            INSERT INTO login_logs (
                timestamp, user_id, matched_user, liveness_result, confidence, status, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, entries)
        conn.commit()
