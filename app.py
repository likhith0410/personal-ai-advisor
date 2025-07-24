# Enhanced Personal AI Advisor Platform
# Secure version with environment variables for API keys

import streamlit as st
import hashlib
import os
import json
import sqlite3
import pandas as pd
import PyPDF2
import docx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid
import re
import requests
import json
import random
import string

# Optional plotting imports
PLOTTING_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Plotly not available - install with: pip install plotly")

# Configuration
USER_DB_PATH = "./users.db"
DOCS_DIR = "./documents"

# Admin Configuration
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # Change this!

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)

def generate_otp() -> str:
    """Generate 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def save_otp_to_session(username_or_email: str, otp: str):
    """Save OTP to session for demo purposes"""
    if "temp_otps" not in st.session_state:
        st.session_state.temp_otps = {}
    
    st.session_state.temp_otps[username_or_email] = {
        "otp": otp,
        "expiry": datetime.now() + timedelta(minutes=10)
    }

# Auto-save API keys in database
def save_api_key(user_id: int, api_key: str):
    """Save API key in database"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # Check if api_key column exists, if not add it
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN api_key TEXT')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Update user's API key
        cursor.execute('UPDATE users SET api_key = ? WHERE id = ?', (api_key, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        pass

def load_api_key(user_id: int) -> str:
    """Load saved API key from database"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT api_key FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return result[0]
    except Exception as e:
        pass
    return ""

def check_and_update_database():
    """Check if database needs updates and migrate if necessary"""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if new columns exist
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add missing columns to users table
        if 'last_login' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN last_login TIMESTAMP')
        if 'total_chats' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN total_chats INTEGER DEFAULT 0')
        if 'total_documents' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN total_documents INTEGER DEFAULT 0')
        if 'is_active' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1')
        if 'subscription_type' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN subscription_type TEXT DEFAULT "free"')
        if 'otp' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN otp TEXT')
        if 'otp_expiry' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN otp_expiry TIMESTAMP')
        if 'api_key' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN api_key TEXT')
        
        # Check advisors table
        cursor.execute("PRAGMA table_info(advisors)")
        advisor_columns = [column[1] for column in cursor.fetchall()]
        
        if 'total_chats' not in advisor_columns:
            cursor.execute('ALTER TABLE advisors ADD COLUMN total_chats INTEGER DEFAULT 0')
        if 'total_documents' not in advisor_columns:
            cursor.execute('ALTER TABLE advisors ADD COLUMN total_documents INTEGER DEFAULT 0')
        if 'last_used' not in advisor_columns:
            cursor.execute('ALTER TABLE advisors ADD COLUMN last_used TIMESTAMP')
        
        # Check documents table
        cursor.execute("PRAGMA table_info(documents)")
        doc_columns = [column[1] for column in cursor.fetchall()]
        
        if 'file_size' not in doc_columns:
            cursor.execute('ALTER TABLE documents ADD COLUMN file_size INTEGER')
        if 'word_count' not in doc_columns:
            cursor.execute('ALTER TABLE documents ADD COLUMN word_count INTEGER')
        if 'summary' not in doc_columns:
            cursor.execute('ALTER TABLE documents ADD COLUMN summary TEXT')
        
        # Check chat_history table
        cursor.execute("PRAGMA table_info(chat_history)")
        chat_columns = [column[1] for column in cursor.fetchall()]
        
        if 'response_time' not in chat_columns:
            cursor.execute('ALTER TABLE chat_history ADD COLUMN response_time REAL')
        if 'rating' not in chat_columns:
            cursor.execute('ALTER TABLE chat_history ADD COLUMN rating INTEGER')
        
        conn.commit()
        print("Database updated successfully")
        
    except Exception as e:
        print(f"Database migration error: {e}")
    finally:
        conn.close()

class UserManager:
    def __init__(self):
        self.init_user_db()
        check_and_update_database()  # Run migration after initialization
    
    def init_user_db(self):
        """Initialize SQLite database with enhanced tables"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # Users table with all fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                total_chats INTEGER DEFAULT 0,
                total_documents INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                subscription_type TEXT DEFAULT 'free',
                otp TEXT,
                otp_expiry TIMESTAMP,
                api_key TEXT
            )
        ''')
        
        # Advisors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advisors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                description TEXT,
                subject_area TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_chats INTEGER DEFAULT 0,
                total_documents INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Documents table with enhanced metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                advisor_id INTEGER,
                filename TEXT NOT NULL,
                content TEXT,
                file_type TEXT,
                file_size INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                summary TEXT,
                FOREIGN KEY (advisor_id) REFERENCES advisors (id)
            )
        ''')
        
        # Chat history with enhanced tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                advisor_id INTEGER,
                user_message TEXT,
                ai_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                rating INTEGER,
                FOREIGN KEY (advisor_id) REFERENCES advisors (id)
            )
        ''')
        
        # Usage analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action_type TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, username: str, email: str, password: str, full_name: str) -> Dict[str, Any]:
        """Register a new user"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name) 
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, full_name))
            
            user_id = cursor.lastrowid
            
            # Log registration
            cursor.execute('''
                INSERT INTO usage_analytics (user_id, action_type, details) 
                VALUES (?, ?, ?)
            ''', (user_id, "registration", f"User {username} registered"))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "user_id": user_id, "message": "Registration successful!"}
        
        except sqlite3.IntegrityError as e:
            conn.close()
            if "username" in str(e):
                return {"success": False, "message": "Username already exists"}
            elif "email" in str(e):
                return {"success": False, "message": "Email already exists"}
            else:
                return {"success": False, "message": "Registration failed"}
    
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return user info"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # First try with is_active column, fallback if it doesn't exist
        try:
            cursor.execute('''
                SELECT id, username, email, full_name, is_active FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            
            if result and (len(result) < 5 or result[4]):  # Check if user is active
                user_id = result[0]
                
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = ? WHERE id = ?
                ''', (datetime.now().isoformat(), user_id))
                
                # Log login
                cursor.execute('''
                    INSERT INTO usage_analytics (user_id, action_type, details) 
                    VALUES (?, ?, ?)
                ''', (user_id, "login", f"User {username} logged in"))
                
                conn.commit()
                conn.close()
                
                return {
                    "authenticated": True,
                    "user_id": result[0],
                    "username": result[1],
                    "email": result[2],
                    "full_name": result[3]
                }
        
        except sqlite3.OperationalError:
            # Fallback for old database structure
            cursor.execute('''
                SELECT id, username, email, full_name FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            
            if result:
                conn.close()
                return {
                    "authenticated": True,
                    "user_id": result[0],
                    "username": result[1],
                    "email": result[2],
                    "full_name": result[3]
                }
        
        conn.close()
        return {"authenticated": False}
    
    def initiate_password_reset(self, username_or_email: str) -> Dict[str, Any]:
        """Initiate password reset process - Demo version without email"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # Find user by username or email
        cursor.execute('''
            SELECT id, email FROM users 
            WHERE username = ? OR email = ?
        ''', (username_or_email, username_or_email))
        
        result = cursor.fetchone()
        
        if result:
            user_id, email = result
            otp = generate_otp()
            otp_expiry = (datetime.now() + timedelta(minutes=10)).isoformat()
            
            # Save OTP to database
            cursor.execute('''
                UPDATE users SET otp = ?, otp_expiry = ? WHERE id = ?
            ''', (otp, otp_expiry, user_id))
            
            conn.commit()
            conn.close()
            
            # Save to session for demo
            save_otp_to_session(username_or_email, otp)
            
            return {
                "success": True, 
                "message": f"Password reset initiated for {email}",
                "demo_otp": otp  # Show OTP for demo - remove in production
            }
        
        conn.close()
        return {"success": False, "message": "User not found"}
    
    def verify_otp_and_reset_password(self, username_or_email: str, otp: str, new_password: str) -> Dict[str, Any]:
        """Verify OTP and reset password"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, otp, otp_expiry FROM users 
            WHERE (username = ? OR email = ?) AND otp = ?
        ''', (username_or_email, username_or_email, otp))
        
        result = cursor.fetchone()
        
        if result:
            user_id, stored_otp, otp_expiry = result
            
            # Check if OTP is expired
            if datetime.now() > datetime.fromisoformat(otp_expiry):
                conn.close()
                return {"success": False, "message": "OTP has expired"}
            
            # Reset password
            new_password_hash = hashlib.sha256(new_password.encode()).hexdigest()
            cursor.execute('''
                UPDATE users SET password_hash = ?, otp = NULL, otp_expiry = NULL 
                WHERE id = ?
            ''', (new_password_hash, user_id))
            
            # Log password reset
            cursor.execute('''
                INSERT INTO usage_analytics (user_id, action_type, details) 
                VALUES (?, ?, ?)
            ''', (user_id, "password_reset", "Password reset successful"))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": "Password reset successful!"}
        
        conn.close()
        return {"success": False, "message": "Invalid OTP"}
    
    def get_admin_stats(self) -> Dict:
        """Get admin dashboard statistics"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total users
        try:
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            stats['total_users'] = cursor.fetchone()[0]
        except:
            cursor.execute('SELECT COUNT(*) FROM users')
            stats['total_users'] = cursor.fetchone()[0]
        
        # Total advisors
        cursor.execute('SELECT COUNT(*) FROM advisors')
        stats['total_advisors'] = cursor.fetchone()[0]
        
        # Total documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        stats['total_documents'] = cursor.fetchone()[0]
        
        # Total chats
        cursor.execute('SELECT COUNT(*) FROM chat_history')
        stats['total_chats'] = cursor.fetchone()[0]
        
        # Recent registrations (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM users WHERE created_at > ?', (week_ago,))
        stats['recent_registrations'] = cursor.fetchone()[0]
        
        # Active users (logged in last 7 days)
        try:
            cursor.execute('SELECT COUNT(*) FROM users WHERE last_login > ?', (week_ago,))
            stats['active_users'] = cursor.fetchone()[0]
        except:
            stats['active_users'] = 0
        
        conn.close()
        return stats
    
    def get_all_users(self) -> List[Dict]:
        """Get all users for admin panel"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, username, email, full_name, created_at, last_login, 
                       total_chats, total_documents, is_active, subscription_type
                FROM users ORDER BY created_at DESC
            ''')
        except sqlite3.OperationalError:
            # Fallback for old database
            cursor.execute('''
                SELECT id, username, email, full_name, created_at
                FROM users ORDER BY created_at DESC
            ''')
        
        users = []
        for row in cursor.fetchall():
            user_dict = {
                "id": row[0],
                "username": row[1],
                "email": row[2],
                "full_name": row[3],
                "created_at": row[4]
            }
            
            # Add optional fields if they exist
            if len(row) > 5:
                user_dict.update({
                    "last_login": row[5],
                    "total_chats": row[6] if len(row) > 6 else 0,
                    "total_documents": row[7] if len(row) > 7 else 0,
                    "is_active": row[8] if len(row) > 8 else True,
                    "subscription_type": row[9] if len(row) > 9 else "free"
                })
            
            users.append(user_dict)
        
        conn.close()
        return users
    
    def create_advisor(self, user_id: int, name: str, description: str, subject_area: str) -> int:
        """Create a new advisor for a user"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO advisors (user_id, name, description, subject_area) 
            VALUES (?, ?, ?, ?)
        ''', (user_id, name, description, subject_area))
        
        advisor_id = cursor.lastrowid
        
        # Log advisor creation
        cursor.execute('''
            INSERT INTO usage_analytics (user_id, action_type, details) 
            VALUES (?, ?, ?)
        ''', (user_id, "advisor_created", f"Created advisor: {name}"))
        
        conn.commit()
        conn.close()
        
        return advisor_id
    
    def get_user_advisors(self, user_id: int) -> List[Dict]:
        """Get all advisors for a user"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, name, description, subject_area, created_at, total_chats, total_documents
                FROM advisors WHERE user_id = ? ORDER BY last_used DESC, created_at DESC
            ''', (user_id,))
        except sqlite3.OperationalError:
            # Fallback for old database
            cursor.execute('''
                SELECT id, name, description, subject_area, created_at
                FROM advisors WHERE user_id = ? ORDER BY created_at DESC
            ''', (user_id,))
        
        advisors = []
        for row in cursor.fetchall():
            advisor_dict = {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "subject_area": row[3],
                "created_at": row[4]
            }
            
            # Add optional fields if they exist
            if len(row) > 5:
                advisor_dict.update({
                    "total_chats": row[5],
                    "total_documents": row[6]
                })
            else:
                advisor_dict.update({
                    "total_chats": 0,
                    "total_documents": 0
                })
            
            advisors.append(advisor_dict)
        
        conn.close()
        return advisors
    
    def delete_advisor(self, advisor_id: int):
        """Delete an advisor and all associated data"""
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # Delete chat history
        cursor.execute('DELETE FROM chat_history WHERE advisor_id = ?', (advisor_id,))
        # Delete documents
        cursor.execute('DELETE FROM documents WHERE advisor_id = ?', (advisor_id,))
        # Delete advisor
        cursor.execute('DELETE FROM advisors WHERE id = ?', (advisor_id,))
        
        conn.commit()
        conn.close()

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error extracting text from DOCX: {str(e)}"
    
    @staticmethod
    def process_file(file) -> str:
        """Process uploaded file and extract text"""
        filename = file.name.lower()
        
        if filename.endswith('.pdf'):
            return DocumentProcessor.extract_text_from_pdf(file)
        elif filename.endswith('.docx'):
            return DocumentProcessor.extract_text_from_docx(file)
        elif filename.endswith('.txt'):
            return str(file.read(), 'utf-8')
        elif filename.endswith('.csv'):
            df = pd.read_csv(file)
            return df.to_string()
        elif filename.endswith('.json'):
            return str(file.read(), 'utf-8')
        else:
            return str(file.read(), 'utf-8')
    
    @staticmethod
    def get_word_count(text: str) -> int:
        """Get word count of text"""
        return len(text.split())
    
    @staticmethod
    def generate_summary(text: str) -> str:
        """Generate a simple summary of the text"""
        sentences = text.split('.')
        # Take first 3 sentences as summary
        summary = '. '.join(sentences[:3])
        return summary[:500] + "..." if len(summary) > 500 else summary

class AdvancedDocumentManager:
    def __init__(self, advisor_id: int):
        self.advisor_id = advisor_id
    
    def add_document(self, content: str, filename: str, file_size: int) -> bool:
        """Add document to database with enhanced metadata"""
        try:
            conn = sqlite3.connect(USER_DB_PATH)
            cursor = conn.cursor()
            
            word_count = DocumentProcessor.get_word_count(content)
            summary = DocumentProcessor.generate_summary(content)
            
            cursor.execute('''
                INSERT INTO documents (advisor_id, filename, content, file_type, 
                                     file_size, upload_date, word_count, summary) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.advisor_id, filename, content, filename.split('.')[-1], 
                  file_size, datetime.now().isoformat(), word_count, summary))
            
            # Update advisor document count
            try:
                cursor.execute('''
                    UPDATE advisors SET total_documents = total_documents + 1 
                    WHERE id = ?
                ''', (self.advisor_id,))
            except sqlite3.OperationalError:
                pass  # Column might not exist in old database
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving document: {str(e)}")
            return False
    
    def search_documents(self, query: str) -> str:
        """Enhanced text search in documents"""
        try:
            conn = sqlite3.connect(USER_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT content, filename FROM documents WHERE advisor_id = ?
            ''', (self.advisor_id,))
            
            documents = cursor.fetchall()
            conn.close()
            
            if not documents:
                return ""
            
            # Enhanced keyword matching
            query_words = set(query.lower().split())
            relevant_content = []
            
            for content, filename in documents:
                if content:
                    # Split into sentences and paragraphs
                    paragraphs = content.split('\n')
                    for paragraph in paragraphs:
                        if len(paragraph.strip()) > 50:  # Only meaningful paragraphs
                            paragraph_words = set(paragraph.lower().split())
                            overlap = len(query_words.intersection(paragraph_words))
                            if overlap > 0:
                                relevant_content.append(paragraph.strip())
            
            # Return best matching content
            return "\n\n".join(relevant_content[:10]) if relevant_content else ""
            
        except Exception as e:
            return ""
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all uploaded documents with metadata"""
        try:
            conn = sqlite3.connect(USER_DB_PATH)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT filename, file_type, file_size, upload_date, word_count, summary
                    FROM documents WHERE advisor_id = ? ORDER BY upload_date DESC
                ''', (self.advisor_id,))
            except sqlite3.OperationalError:
                # Fallback for old database
                cursor.execute('''
                    SELECT filename, file_type, upload_date
                    FROM documents WHERE advisor_id = ? ORDER BY upload_date DESC
                ''', (self.advisor_id,))
            
            documents = []
            for row in cursor.fetchall():
                doc_dict = {
                    "filename": row[0],
                    "file_type": row[1] if len(row) > 1 else "unknown",
                    "upload_date": row[2] if len(row) > 2 else "unknown"
                }
                
                # Add optional fields if they exist
                if len(row) > 3:
                    doc_dict.update({
                        "file_size": row[2] if len(row) > 2 else 0,
                        "upload_date": row[3] if len(row) > 3 else "unknown",
                        "word_count": row[4] if len(row) > 4 else 0,
                        "summary": row[5] if len(row) > 5 else ""
                    })
                else:
                    doc_dict.update({
                        "file_size": 0,
                        "word_count": 0,
                        "summary": ""
                    })
                
                documents.append(doc_dict)
            
            conn.close()
            return documents
        except Exception as e:
            return []
    
    def delete_document(self, filename: str) -> bool:
        """Delete a document"""
        try:
            conn = sqlite3.connect(USER_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM documents WHERE advisor_id = ? AND filename = ?
            ''', (self.advisor_id, filename))
            
            # Update advisor document count
            try:
                cursor.execute('''
                    UPDATE advisors SET total_documents = total_documents - 1 
                    WHERE id = ?
                ''', (self.advisor_id,))
            except sqlite3.OperationalError:
                pass  # Column might not exist in old database
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            return False

class IntelligentAI:
    def __init__(self, user_id: int):
        self.user_id = user_id
        
        # Try to get API key in this order:
        # 1. User's custom key (if they set one)
        # 2. Environment variable (for deployment)
        # 3. Streamlit secrets (for Streamlit Cloud)
        
        user_key = load_api_key(user_id)
        if user_key:
            self.groq_api_key = user_key
            self.api_source = "user_custom"
        else:
            # Try environment variable
            self.groq_api_key = os.getenv('GROQ_API_KEY')
            if self.groq_api_key:
                self.api_source = "environment"
            else:
                # Try Streamlit secrets if no env var
                try:
                    self.groq_api_key = st.secrets.get("GROQ_API_KEY")
                    if self.groq_api_key:
                        self.api_source = "streamlit_secrets"
                    else:
                        self.api_source = "none"
                except:
                    self.groq_api_key = None
                    self.api_source = "none"
        
    def generate_with_groq(self, prompt: str) -> str:
        """Generate response using Groq API"""
        if not self.groq_api_key:
            return None  # Will use fallback
        
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are an intelligent AI advisor. Provide detailed, helpful, and professional responses based on the context provided. Be conversational and engaging."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 800
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            pass  # Silently fall back to enhanced response
        
        return None
    
    def generate_intelligent_response(self, query: str, context: str, advisor_name: str) -> str:
        """Generate intelligent AI response"""
        
        # Create comprehensive prompt
        if context.strip():
            prompt = f"""As {advisor_name}, an expert AI advisor, analyze the following context and answer the user's question comprehensively.

Context from uploaded documents:
{context}

User's Question: {query}

Provide a detailed, professional response as {advisor_name}. Include specific insights, recommendations, and actionable advice based on the context."""
        else:
            prompt = f"""As {advisor_name}, an expert AI advisor, the user is asking: {query}

Provide a helpful response explaining that you need them to upload relevant documents first so you can give them specific, detailed advice based on their materials."""
        
        # Try Groq API first if available
        if self.groq_api_key:
            ai_response = self.generate_with_groq(prompt)
            if ai_response:
                return ai_response
        
        # Enhanced fallback if no API key or API fails
        return self.enhanced_fallback(query, context, advisor_name)
    
    def enhanced_fallback(self, query: str, context: str, advisor_name: str) -> str:
        """Enhanced fallback response"""
        if not context.strip():
            # Professional responses for common queries without documents
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
                return f"""Hello! I'm {advisor_name}, your professional AI advisor. I'm here to provide you with expert insights and recommendations. 

To give you the most accurate and personalized advice, I'd recommend uploading relevant documents such as:
• Reports, presentations, or research materials
• Policy documents or guidelines  
• Data files or spreadsheets
• Any other materials related to your inquiry

Once you provide these materials, I can analyze them and offer detailed, context-specific guidance. How can I assist you today?"""
            
            elif any(word in query_lower for word in ['help', 'assist', 'support', 'advice']):
                return f"""I'm {advisor_name}, and I'm here to provide professional consultation and strategic advice. My expertise spans across various domains, and I can help you with:

• Strategic analysis and recommendations
• Document review and insights
• Data interpretation and trends
• Best practices and methodologies
• Problem-solving approaches

To provide you with the most valuable insights, please upload any relevant documents or materials related to your specific needs. This will allow me to deliver targeted, actionable advice tailored to your situation."""
            
            elif any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
                return f"""Thank you for reaching out. As {advisor_name}, I specialize in providing comprehensive analysis and strategic guidance. 

To address your question effectively, I'll need access to relevant background materials or documentation. This could include:
• Project specifications or requirements
• Current processes or procedures
• Data sets or performance metrics
• Industry reports or benchmarks

Please upload any pertinent documents, and I'll provide you with detailed insights, recommendations, and actionable solutions specific to your inquiry."""
            
            else:
                return f"""Welcome! I'm {advisor_name}, your dedicated AI advisor ready to provide expert analysis and strategic recommendations.

I notice you haven't uploaded any supporting documents yet. To deliver the most valuable and accurate insights for your specific situation, please share relevant materials such as:
• Business documents or reports
• Technical specifications
• Data files or analytics
• Reference materials or guidelines

Once I have access to your materials, I can provide comprehensive analysis, identify key insights, and offer actionable recommendations tailored to your needs."""
        
        # Analyze context intelligently for document-based responses
        sentences = [s.strip() for s in context.split('.') if s.strip() and len(s.strip()) > 30]
        query_words = set(query.lower().split())
        
        # Find most relevant information
        relevant_info = []
        for sentence in sentences[:15]:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) > 0:
                relevant_info.append(sentence)
        
        if relevant_info:
            response = f"""Based on my analysis of your uploaded documents, here are the key insights regarding your inquiry:

"""
            
            for i, info in enumerate(relevant_info[:4], 1):
                response += f"**{i}. Key Finding:** {info}.\n\n"
            
            response += f"""**Recommendations:**
Based on this analysis, I recommend further exploration of the specific aspects that align with your objectives. 

Would you like me to elaborate on any particular finding, or do you have additional questions about the implementation strategies for these insights?"""
        else:
            response = f"""I've reviewed your uploaded materials and understand you're inquiring about this topic. While I can see the documentation you've provided, I'd benefit from a more specific question to deliver the most targeted analysis.

**Suggested approach:**
• Ask about specific sections or concepts from your documents
• Request analysis of particular data points or trends
• Inquire about implementation strategies or best practices
• Seek recommendations for specific challenges or opportunities

This will allow me to provide more focused, actionable insights from your materials."""
        
        return response

class ConversationManager:
    @staticmethod
    def save_chat(advisor_id: int, user_message: str, ai_response: str, response_time: float):
        """Save chat to database"""
        try:
            conn = sqlite3.connect(USER_DB_PATH)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO chat_history (advisor_id, user_message, ai_response, response_time) 
                    VALUES (?, ?, ?, ?)
                ''', (advisor_id, user_message, ai_response, response_time))
            except sqlite3.OperationalError:
                # Fallback for old database
                cursor.execute('''
                    INSERT INTO chat_history (advisor_id, user_message, ai_response) 
                    VALUES (?, ?, ?)
                ''', (advisor_id, user_message, ai_response))
            
            # Update advisor chat count
            try:
                cursor.execute('''
                    UPDATE advisors SET total_chats = total_chats + 1, last_used = ? 
                    WHERE id = ?
                ''', (datetime.now().isoformat(), advisor_id))
            except sqlite3.OperationalError:
                pass  # Columns might not exist in old database
            
            conn.commit()
            conn.close()
        except Exception as e:
            pass
    
    @staticmethod
    def export_conversation(advisor_id: int, format: str = "json") -> str:
        """Export conversation history"""
        try:
            conn = sqlite3.connect(USER_DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, ai_response, timestamp 
                FROM chat_history WHERE advisor_id = ? ORDER BY timestamp
            ''', (advisor_id,))
            
            chats = []
            for row in cursor.fetchall():
                chats.append({
                    "user_message": row[0],
                    "ai_response": row[1],
                    "timestamp": row[2]
                })
            
            conn.close()
            
            if format == "json":
                return json.dumps(chats, indent=2)
            elif format == "txt":
                text = ""
                for chat in chats:
                    text += f"[{chat['timestamp']}]\n"
                    text += f"User: {chat['user_message']}\n"
                    text += f"AI: {chat['ai_response']}\n\n"
                return text
            
        except Exception as e:
            return f"Error exporting: {str(e)}"

def apply_futuristic_theme():
    """Apply advanced futuristic theme"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@200;300;400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-bg: #0a0a0f;
        --secondary-bg: #1a1a2e;
        --accent-bg: #16213e;
        --primary-text: #e6e6e6;
        --secondary-text: #b3b3b3;
        --accent-color: #00d4ff;
        --success-color: #00ff88;
        --warning-color: #ffaa00;
        --border-color: #2a2a3e;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        --gradient-dark: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%);
    }
    
    /* Global styles */
    .stApp {
        background: var(--gradient-dark);
        color: var(--primary-text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stActionButton {display: none;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    
    .main-header h1 {
        background: var(--gradient-secondary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: var(--secondary-text);
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Enhanced cards */
    .advisor-card, .metric-card, .admin-card {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .advisor-card::before, .metric-card::before, .admin-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-secondary);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .advisor-card:hover::before, .metric-card:hover::before, .admin-card:hover::before {
        transform: scaleX(1);
    }
    
    .advisor-card:hover, .metric-card:hover, .admin-card:hover {
        transform: translateY(-4px);
        border-color: var(--accent-color);
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.15);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        filter: brightness(1.1);
    }
    
    /* Enhanced inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--primary-text);
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-color);
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2);
    }
    
    /* Enhanced file uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.5) 0%, rgba(22, 33, 62, 0.5) 100%);
        border: 2px dashed var(--accent-color);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    /* High contrast text for visibility */
    .stMarkdown, .stText, p, span, div {
        color: var(--primary-text) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--accent-color) !important;
    }
    
    /* Tables */
    .stDataFrame {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-color);
    }
    
    .metric-label {
        color: var(--secondary-text);
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def login_register_page():
    """Enhanced login and registration page with password reset"""
    apply_futuristic_theme()
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Advisor Platform</h1>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔑 Login", "📝 Register", "🔄 Reset Password"])
    
    with tab1:
        st.markdown("### Welcome Back!")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                user_manager = UserManager()
                auth_result = user_manager.authenticate(username, password)
                
                if auth_result["authenticated"]:
                    st.session_state.user = auth_result
                    st.success("Welcome back! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        st.markdown("### Create Your Account")
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input("Username*")
                reg_email = st.text_input("Email*")
            
            with col2:
                reg_full_name = st.text_input("Full Name*")
                reg_password = st.text_input("Password*", type="password")
            
            reg_submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if reg_submit:
                if all([reg_username, reg_email, reg_full_name, reg_password]):
                    user_manager = UserManager()
                    result = user_manager.register_user(reg_username, reg_email, reg_password, reg_full_name)
                    
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
                else:
                    st.error("Please fill in all required fields")
    
    with tab3:
        st.markdown("### Reset Your Password")
        st.info("🚀 **Demo Mode**: Password reset works without email configuration!")
        
        if "reset_step" not in st.session_state:
            st.session_state.reset_step = 1
        
        if st.session_state.reset_step == 1:
            # Step 1: Request OTP
            with st.form("reset_request_form"):
                st.markdown("**Step 1:** Enter your username or email")
                username_or_email = st.text_input("Username or Email")
                
                if st.form_submit_button("Generate OTP", use_container_width=True):
                    if username_or_email:
                        user_manager = UserManager()
                        result = user_manager.initiate_password_reset(username_or_email)
                        
                        if result["success"]:
                            st.success(result["message"])
                            st.info(f"🔓 **Your OTP is:** `{result['demo_otp']}`")
                            st.session_state.reset_username_or_email = username_or_email
                            st.session_state.reset_step = 2
                            st.rerun()
                        else:
                            st.error(result["message"])
                    else:
                        st.error("Please enter username or email")
        
        elif st.session_state.reset_step == 2:
            # Step 2: Verify OTP and reset password
            with st.form("reset_verify_form"):
                st.markdown("**Step 2:** Enter OTP and new password")
                otp = st.text_input("OTP (6 digits)")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.form_submit_button("Reset Password", use_container_width=True):
                        if otp and new_password and confirm_password:
                            if new_password == confirm_password:
                                user_manager = UserManager()
                                result = user_manager.verify_otp_and_reset_password(
                                    st.session_state.reset_username_or_email, otp, new_password
                                )
                                
                                if result["success"]:
                                    st.success(result["message"])
                                    st.session_state.reset_step = 1
                                    del st.session_state.reset_username_or_email
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                            else:
                                st.error("Passwords don't match")
                        else:
                            st.error("Please fill all fields")
                
                with col2:
                    if st.form_submit_button("Back", use_container_width=True):
                        st.session_state.reset_step = 1
                        st.rerun()

def admin_panel():
    """Complete admin panel"""
    apply_futuristic_theme()
    
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Admin Dashboard</h1>
        <p>Monitor and manage your AI platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_manager = UserManager()
    
    # Admin stats
    stats = user_manager.get_admin_stats()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Users</div>
        </div>
        """.format(stats['total_users']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Advisors</div>
        </div>
        """.format(stats['total_advisors']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Documents</div>
        </div>
        """.format(stats['total_documents']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Chats</div>
        </div>
        """.format(stats['total_chats']), unsafe_allow_html=True)
    
    # Tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["👥 Users", "📊 Analytics", "💾 Data Export"])
    
    with tab1:
        st.markdown("### User Management")
        
        users = user_manager.get_all_users()
        
        if users:
            # Create DataFrame
            df = pd.DataFrame(users)
            st.dataframe(df, use_container_width=True)
            
            # User actions
            st.markdown("### User Actions")
            selected_user = st.selectbox("Select User", [f"{u['username']} ({u['email']})" for u in users])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("View Details"):
                    # Show user details
                    user = next(u for u in users if f"{u['username']} ({u['email']})" == selected_user)
                    st.json(user)
            
            with col2:
                if st.button("Deactivate"):
                    st.warning("Feature coming soon")
            
            with col3:
                if st.button("Send Message"):
                    st.warning("Feature coming soon")
    
    with tab2:
        st.markdown("### Platform Analytics")
        
        # Simple analytics without plotly dependency
        conn = sqlite3.connect(USER_DB_PATH)
        
        # Daily registrations
        try:
            query = """
            SELECT DATE(created_at) as date, COUNT(*) as registrations
            FROM users 
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY date
            """
            df_reg = pd.read_sql_query(query, conn)
            
            if not df_reg.empty:
                st.markdown("### Daily Registrations (Last 30 Days)")
                if PLOTTING_AVAILABLE:
                    fig = px.line(df_reg, x='date', y='registrations', title='Daily Registrations')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df_reg, use_container_width=True)
            else:
                st.info("No registration data available yet")
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
        
        # Usage analytics
        try:
            query = """
            SELECT action_type, COUNT(*) as count
            FROM usage_analytics
            WHERE timestamp >= date('now', '-7 days')
            GROUP BY action_type
            """
            df_usage = pd.read_sql_query(query, conn)
            
            if not df_usage.empty:
                st.markdown("### User Actions (Last 7 Days)")
                if PLOTTING_AVAILABLE:
                    fig = px.pie(df_usage, values='count', names='action_type', title='User Actions')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(df_usage, use_container_width=True)
            else:
                st.info("No usage data available yet")
        except Exception as e:
            st.info("Usage analytics table not available - will be created as users interact with the platform")
        
        conn.close()
    
    with tab3:
        st.markdown("### Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Users CSV"):
                users = user_manager.get_all_users()
                df = pd.DataFrame(users)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Users CSV",
                    data=csv,
                    file_name=f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Analytics JSON"):
                try:
                    conn = sqlite3.connect(USER_DB_PATH)
                    df = pd.read_sql_query("SELECT * FROM usage_analytics", conn)
                    conn.close()
                    
                    json_data = df.to_json(orient='records')
                    st.download_button(
                        label="Download Analytics JSON",
                        data=json_data,
                        file_name=f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error("No analytics data available yet")

def advisor_management():
    """Enhanced advisor management interface"""
    apply_futuristic_theme()
    
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Your AI Advisors</h1>
        <p>Create and manage your intelligent AI advisors</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_manager = UserManager()
    advisors = user_manager.get_user_advisors(st.session_state.user["user_id"])
    
    # Sidebar for creating new advisor
    with st.sidebar:
        st.markdown("### ➕ Create New Advisor")
        
        with st.form("create_advisor"):
            advisor_name = st.text_input("Advisor Name*", placeholder="e.g., Career Counselor")
            subject_area = st.selectbox(
                "Subject Area*",
                ["Finance & Investment", "Career & HR", "Medical & Health", "Legal", "Technology", 
                 "Education", "Marketing", "Science", "Business", "Personal Development", "Other"]
            )
            description = st.text_area("Description", placeholder="What this advisor specializes in...")
            
            if st.form_submit_button("Create Advisor", use_container_width=True):
                if advisor_name and subject_area:
                    advisor_id = user_manager.create_advisor(
                        st.session_state.user["user_id"],
                        advisor_name,
                        description,
                        subject_area
                    )
                    st.success(f"✅ {advisor_name} created!")
                    st.rerun()
                else:
                    st.error("Please fill in required fields")
    
    # Display existing advisors
    if advisors:
        st.markdown("### Your Advisors")
        
        for advisor in advisors:
            st.markdown(f"""
            <div class="advisor-card">
                <h3>🤖 {advisor['name']}</h3>
                <p><strong>Subject:</strong> {advisor['subject_area']}</p>
                <p><strong>Description:</strong> {advisor['description'] or 'General AI advisor'}</p>
                <p><strong>Stats:</strong> {advisor['total_chats']} chats • {advisor['total_documents']} documents</p>
                <p><small>Created: {advisor['created_at'][:10]}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💬 Chat", key=f"chat_{advisor['id']}"):
                    st.session_state.current_advisor = advisor['id']
                    st.session_state.page = "chat"
                    st.rerun()
            
            with col2:
                if st.button("📁 Manage", key=f"manage_{advisor['id']}"):
                    st.session_state.current_advisor = advisor['id']
                    st.session_state.page = "manage_documents"
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{advisor['id']}"):
                    user_manager.delete_advisor(advisor['id'])
                    st.success(f"Deleted {advisor['name']}")
                    st.rerun()
    else:
        st.info("No advisors yet. Create your first AI advisor using the sidebar!")

def document_management():
    """Enhanced document management"""
    apply_futuristic_theme()
    
    advisor_id = st.session_state.current_advisor
    user_manager = UserManager()
    advisors = user_manager.get_user_advisors(st.session_state.user["user_id"])
    current_advisor = next((a for a in advisors if a["id"] == advisor_id), None)
    
    if not current_advisor:
        st.error("Advisor not found!")
        return
    
    st.markdown(f"""
    <div class="main-header">
        <h1>📚 Document Management</h1>
        <p>Training materials for {current_advisor['name']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    doc_manager = AdvancedDocumentManager(advisor_id)
    
    # File upload section
    st.markdown("### 📤 Upload Training Documents")
    
    uploaded_files = st.file_uploader(
        "Upload documents to train your advisor",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'csv', 'json'],
        help="Supported: PDF, DOCX, TXT, CSV, JSON"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    content = DocumentProcessor.process_file(uploaded_file)
                    
                    if content.strip():
                        if doc_manager.add_document(content, uploaded_file.name, uploaded_file.size):
                            st.success(f"✅ {uploaded_file.name} processed!")
                        else:
                            st.error(f"❌ Failed to save {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ No text found in {uploaded_file.name}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Enhanced document display
    st.markdown("### 📄 Uploaded Documents")
    documents = doc_manager.get_all_documents()
    
    if documents:
        st.write(f"**Total Documents:** {len(documents)}")
        
        # Document statistics
        total_words = sum(doc.get('word_count', 0) for doc in documents)
        total_size = sum(doc.get('file_size', 0) for doc in documents)
        
        if total_words > 0 or total_size > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", f"{total_words:,}")
            with col2:
                st.metric("Total Size", f"{total_size / 1024:.1f} KB" if total_size > 0 else "N/A")
            with col3:
                st.metric("Avg Words/Doc", f"{total_words // len(documents):,}" if total_words > 0 else "N/A")
        
        # Document list with enhanced info
        for i, doc in enumerate(documents):
            with st.expander(f"📄 {doc['filename']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Type:** {doc['file_type'].upper()}")
                    if doc.get('file_size', 0) > 0:
                        st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    if doc.get('word_count', 0) > 0:
                        st.write(f"**Words:** {doc['word_count']:,}")
                    st.write(f"**Uploaded:** {doc['upload_date'][:10]}")
                    
                    if doc.get('summary'):
                        st.write(f"**Summary:** {doc['summary']}")
                
                with col2:
                    unique_key = f"del_{advisor_id}_{i}_{hash(doc['filename']) % 10000}"
                    if st.button("🗑️ Delete", key=unique_key):
                        if doc_manager.delete_document(doc['filename']):
                            st.success(f"Deleted {doc['filename']}")
                            st.rerun()
    else:
        st.info("No documents uploaded yet.")
    
    if st.button("← Back to Advisors"):
        st.session_state.page = "advisors"
        st.rerun()

def chat_interface():
    """Enhanced chat interface with advanced features"""
    apply_futuristic_theme()
    
    advisor_id = st.session_state.current_advisor
    user_manager = UserManager()
    advisors = user_manager.get_user_advisors(st.session_state.user["user_id"])
    current_advisor = next((a for a in advisors if a["id"] == advisor_id), None)
    
    if not current_advisor:
        st.error("Advisor not found!")
        return
    
    st.markdown(f"""
    <div class="main-header">
        <h1>💬 Chat with {current_advisor['name']}</h1>
        <p>Your intelligent {current_advisor['subject_area']} advisor</p>
    </div>
    """, unsafe_allow_html=True)
    
    doc_manager = AdvancedDocumentManager(advisor_id)
    
    # Re-initialize AI object to pick up any new API keys
    ai = IntelligentAI(st.session_state.user["user_id"])
    
    # Show current API status
    if ai.groq_api_key:
        st.success("🤖 AI-powered responses active")
    else:
        st.info("📝 Enhanced response mode active. Add API key in sidebar for AI responses.")
    
    # Initialize chat history
    if f"messages_{advisor_id}" not in st.session_state:
        st.session_state[f"messages_{advisor_id}"] = [
            {"role": "assistant", "content": f"Hello! I'm {current_advisor['name']}, your AI advisor specializing in {current_advisor['subject_area']}. How can I help you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state[f"messages_{advisor_id}"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 📄 Available Documents")
        documents = doc_manager.get_all_documents()
        
        if documents:
            for doc in documents:
                word_count = doc.get('word_count', 0)
                if word_count > 0:
                    st.write(f"• {doc['filename']} ({word_count:,} words)")
                else:
                    st.write(f"• {doc['filename']}")
        else:
            st.info("No documents uploaded yet.")
        
        st.markdown("### 📊 Chat Stats")
        st.write(f"**Total Chats:** {current_advisor['total_chats']}")
        st.write(f"**Documents:** {current_advisor['total_documents']}")
        
        st.markdown("### 🛠️ Chat Tools")
        
        if st.button("📥 Export Chat"):
            export_format = st.selectbox("Format", ["JSON", "TXT"])
            exported = ConversationManager.export_conversation(
                advisor_id, export_format.lower()
            )
            
            st.download_button(
                label=f"Download {export_format}",
                data=exported,
                file_name=f"chat_{current_advisor['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                mime="application/json" if export_format == "JSON" else "text/plain"
            )
        
        if st.button("🗑️ Clear Chat"):
            st.session_state[f"messages_{advisor_id}"] = [
                {"role": "assistant", "content": f"Hello! I'm {current_advisor['name']}, your AI advisor. How can I help you?"}
            ]
            st.rerun()
        
        if st.button("← Back to Advisors"):
            st.session_state.page = "advisors"
            st.rerun()
    
    # Chat input with timing
    if prompt := st.chat_input(f"Ask {current_advisor['name']} anything..."):
        start_time = datetime.now()
        
        # Add user message
        st.session_state[f"messages_{advisor_id}"].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner(f"{current_advisor['name']} is analyzing..."):
                # Search for relevant context
                context = doc_manager.search_documents(prompt)
                
                # Generate intelligent AI response
                response = ai.generate_intelligent_response(prompt, context, current_advisor['name'])
                
                st.markdown(response)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Save chat to database
        ConversationManager.save_chat(advisor_id, prompt, response, response_time)
        
        # Add assistant response
        st.session_state[f"messages_{advisor_id}"].append({"role": "assistant", "content": response})

def main():
    """Enhanced main application function"""
    st.set_page_config(
        page_title="AI Advisor Platform",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "advisors"
    if "current_advisor" not in st.session_state:
        st.session_state.current_advisor = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    
    # Check for admin login
    if "admin" in st.query_params and st.query_params["admin"] == "true":
        if not st.session_state.is_admin:
            st.markdown("### Admin Login")
            admin_username = st.text_input("Admin Username")
            admin_password = st.text_input("Admin Password", type="password")
            
            if st.button("Admin Login"):
                if admin_username == ADMIN_USERNAME and admin_password == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("Admin access granted!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
            
            st.markdown("---")
            st.markdown("**Regular users:** Continue to normal login below")
        else:
            admin_panel()
            return
    
    # Check if user is logged in
    if st.session_state.user is None:
        login_register_page()
    else:
        # Regular user interface
        with st.sidebar:
            st.markdown(f"### 👋 Welcome, {st.session_state.user['full_name']}!")
            st.markdown(f"**@{st.session_state.user['username']}**")
            
            # API Key Setup (Optional)
            with st.expander("🔑 Setup Your API Key (Optional)", expanded=False):
                st.markdown("### Get Your Free Groq API Key")
                st.markdown("**Step 1:** [Click here to create Groq API Key](https://console.groq.com)")
                st.markdown("**Step 2:** Click 'Create API Key'")
                st.markdown("**Step 3:** Copy your API key")
                st.markdown("**Step 4:** Paste it below:")
                
                # Load current API key
                current_key = load_api_key(st.session_state.user["user_id"])
                
                with st.form("api_key_form"):
                    api_key_input = st.text_input(
                        "Your Groq API Key", 
                        value=current_key,
                        type="password",
                        placeholder="gsk_..."
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Save Key"):
                            if api_key_input.strip():
                                save_api_key(st.session_state.user["user_id"], api_key_input.strip())
                                st.success("✅ API Key saved!")
                                st.rerun()
                            else:
                                st.error("Please enter a valid API key")
                    
                    with col2:
                        if st.form_submit_button("Remove Key"):
                            save_api_key(st.session_state.user["user_id"], "")
                            st.success("🗑️ API Key removed!")
                            st.rerun()
                
                # Show current status
                env_key = os.getenv('GROQ_API_KEY')
                try:
                    secrets_key = st.secrets.get("GROQ_API_KEY")
                except:
                    secrets_key = None
                
                if current_key:
                    st.success("🔑 Your custom API key is active")
                elif env_key or secrets_key:
                    st.success("🔑 Platform API key is active")
                else:
                    st.info("🔑 No API key configured - using enhanced fallback mode")
            
            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Route to appropriate page
        if st.session_state.page == "advisors":
            advisor_management()
        elif st.session_state.page == "manage_documents":
            document_management()
        elif st.session_state.page == "chat":
            chat_interface()

if __name__ == "__main__":
    main()
