# firestore_service.py
import os
from google.cloud import firestore
from firebase_admin import credentials, firestore, initialize_app

# Initialize Firebase
cred = credentials.Certificate("ktubot-c3495-firebase-adminsdk-fbsvc-5ec25bbfae.json")
firebase_app = initialize_app(cred)
db = firestore.client()

def get_user_chat_ref(uid: str):
    """Get reference to user's chat collection with existence check"""
    user_ref = db.collection("users").document(uid)
    
    # Create user document if doesn't exist
    if not user_ref.get().exists:
        user_ref.set({
            'created_at': firestore.SERVER_TIMESTAMP,
            'email': f"user_{uid}@temp.com"  # Temporary email for reference
        })
    
    return user_ref.collection("chats")

def save_chat(uid: str, chat_data: dict):
    """Save chat with automatic timestamp and validation"""
    chat_ref = get_user_chat_ref(uid).document()
    
    # Add required fields
    chat_data.update({
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
        'uid': uid
    })
    
    # Set the document
    chat_ref.set(chat_data)
    return chat_ref.id

def get_user_chats(uid: str, limit=20):
    """Get user's chats ordered by timestamp"""
    chats_ref = get_user_chat_ref(uid)
    docs = chats_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit).stream()
    return [doc.to_dict() for doc in docs]