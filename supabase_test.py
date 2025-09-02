import streamlit as st
from supabase import create_client, Client
import io
import json
from datetime import datetime
import uuid
from PIL import Image
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client - works both locally and deployed"""
    try:
        # For deployed app
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except:
        # For local development - add these to your .env file
        import os
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
    
    return create_client(url, key)

supabase: Client = init_supabase()

# Functions for your experiment
def save_participant_data(participant_id, responses_data):
    """Save JSON/CSV data to database"""
    try:
        result = supabase.table('participants').insert({
            'participant_id': participant_id,
            'responses': responses_data,  # JSON data
            'created_at': datetime.now().isoformat(),
            'completed': True
        }).execute()
        return True, result
    except Exception as e:
        return False, str(e)

def save_image(participant_id, image_data, image_name):
    """Save PNG image to storage"""
    try:
        # Convert PIL Image to bytes if needed
        if hasattr(image_data, 'save'):  # PIL Image
            img_bytes = io.BytesIO()
            image_data.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
        else:
            img_bytes = image_data  # Already bytes
        
        # Upload to storage
        file_path = f"{participant_id}/{image_name}.png"
        result = supabase.storage.from_('experiment-images').upload(
            file_path, img_bytes
        )
        return True, result
    except Exception as e:
        return False, str(e)

def save_experiment_complete(participant_id, all_responses, image_list):
    """Save everything for one participant"""
    results = []
    
    # Save responses to database
    success, result = save_participant_data(participant_id, all_responses)
    results.append(("Database", success, result))
    
    # Save all images
    for i, image in enumerate(image_list):
        success, result = save_image(participant_id, image, f"image_{i}")
        results.append((f"Image {i}", success, result))
    
    return results

# Demo/Test interface
st.title("Supabase Integration Test")

st.write("**Storage calculation:**")
st.write(f"- 10 participants × 30MB = 300MB total")
st.write(f"- Supabase free tier: 1GB storage ✅")
st.write(f"- Cost: FREE")

# Test data save
st.subheader("Test Data Save")

if st.button("Test Save Data"):
    # Generate test data
    test_participant_id = str(uuid.uuid4())
    test_responses = {
        "age": 25,
        "responses": [1, 2, 3, 4, 5],
        "completion_time": "5 minutes",
        "feedback": "Great experiment!"
    }
    
    with st.spinner("Saving test data..."):
        success, result = save_participant_data(test_participant_id, test_responses)
    
    if success:
        st.success("✅ Data saved successfully!")
        st.write(f"**Participant ID:** {test_participant_id}")
    else:
        st.error("❌ Failed to save data")
        st.write(f"**Error:** {result}")

# Test image upload
st.subheader("Test Image Upload")

uploaded_file = st.file_uploader("Upload test image", type=['png', 'jpg', 'jpeg'])

if uploaded_file and st.button("Test Save Image"):
    test_participant_id = str(uuid.uuid4())
    
    with st.spinner("Uploading image..."):
        success, result = save_image(test_participant_id, uploaded_file.getvalue(), "test_image")
    
    if success:
        st.success("✅ Image uploaded successfully!")
        st.write(f"**Participant ID:** {test_participant_id}")
    else:
        st.error("❌ Failed to upload image")
        st.write(f"**Error:** {result}")

# Setup instructions
st.write("---")
st.subheader("Setup Instructions")

with st.expander("3. Create database table"):
    st.code("""
-- Go to SQL Editor in Supabase and run:
CREATE TABLE participants (
    id SERIAL PRIMARY KEY,
    participant_id TEXT UNIQUE NOT NULL,
    responses JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed BOOLEAN DEFAULT FALSE
);
    """, language="sql")

with st.expander("4. Create storage bucket"):
    st.write("1. Go to Storage in Supabase dashboard")
    st.write("2. Create New Bucket → Name: `experiment-images`")
    st.write("3. Set to Public (so you can view images later)")



st.write("---")
st.success("🚀 **Total setup time: ~5 minutes**")
st.write("✅ Free tier: 1GB storage (enough for your 300MB)")
st.write("✅ Works perfectly with Streamlit Cloud")
st.write("✅ Easy data export (CSV/JSON)")
st.write("✅ No authentication headaches")