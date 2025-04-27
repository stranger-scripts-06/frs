#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py - Streamlit Fashion Recommendation System
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image as PILImage
import os

# -------------------------------------------
# Setup
# -------------------------------------------

st.set_page_config(page_title="Fashion Recommendation App", page_icon="👗")

st.title("👗 Personalized Fashion Recommendation System")
st.write("Upload your face photo, and we'll suggest outfits and accessories just for you!")

# -------------------------------------------
# Load Dataset
# -------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv('style.csv')
    df = df.dropna(subset=['gender', 'masterCategory', 'baseColour', 'usage'])
    return df

fashion_data = load_data()

# -------------------------------------------
# Color Theory Setup
# -------------------------------------------

color_wheel = {
    'Navy Blue': ['Turquoise Blue', 'Olive'],
    'Black': ['Silver', 'White'],
    'White': ['Pastel Blue', 'Peach'],
    'Red': ['Olive', 'Peach'],
    'Green': ['Red', 'Yellow'],
    'Yellow': ['Olive', 'Sea Green'],
    'Olive': ['Coral', 'Rust'],
    'Peach': ['Turquoise Blue', 'White'],
    'Turquoise Blue': ['Navy Blue', 'Peach'],
    'Coral': ['Olive', 'Sea Green']
}

skin_tone_to_color_palette = {
    'Deep': ['Navy Blue', 'Olive', 'Rust', 'Burgundy', 'Gold'],
    'Tan': ['Coral', 'Sea Green', 'Turquoise Blue', 'Yellow', 'Peach'],
    'Fair': ['Lavender', 'Soft Pink', 'Light Blue', 'Mint Green', 'White'],
    'Pale': ['Off White', 'Silver', 'Pastel Blue', 'Light Grey']
}

occasion_accessories = {
    'Casual': ['Backpack', 'Sunglasses', 'Sneakers', 'Watch'],
    'Formal': ['Leather Belt', 'Wrist Watch', 'Formal Shoes', 'Minimalist Tie'],
    'Party': ['Statement Necklace', 'Clutch Bag', 'Heels', 'Dangly Earrings'],
    'Ethnic': ['Bangles', 'Ethnic Dupatta', 'Kolhapuri Chappals', 'Bindis'],
    'Sports': ['Sports Watch', 'Cap', 'Training Shoes', 'Sweatbands'],
    'Travel': ['Duffel Bag', 'Comfortable Sneakers', 'Crossbody Bag', 'Travel Hat']
}

outfit_images = {
    't-shirt': 'images/tshirt.jpg',
    'shirt': 'images/shirt.jpg',
    'blouse': 'images/blouse.jpg',
    'jeans': 'images/jeans.jpg',
    'trousers': 'images/trousers.jpg',
    'shorts': 'images/shorts.jpg',
    'sneakers': 'images/sneakers.jpg',
    'formal shoes': 'images/formal_shoes.jpg',
    'sandals': 'images/sandals.jpg',
    'flip flops': 'images/flip_flops.jpg'
}

# -------------------------------------------
# Functions
# -------------------------------------------

def closest_skin_tone(avg_rgb):
    ref_tones = {
        'Deep': np.array([66, 37, 17]),
        'Tan': np.array([157, 122, 84]),
        'Fair': np.array([229, 200, 166]),
        'Pale': np.array([243, 240, 240])
    }
    min_dist = float('inf')
    closest = None
    for tone, ref_rgb in ref_tones.items():
        dist = np.linalg.norm(avg_rgb - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest = tone
    return closest

def recommend_outfits(gender, occasion, skin_tone):
    recommended_colors = skin_tone_to_color_palette.get(skin_tone, [])
    
    occasion_mapping = {
        'Casual': ['Casual', 'Travel', 'Sports'],
        'Formal': ['Formal'],
        'Party': ['Party'],
        'Ethnic': ['Ethnic'],
        'Sports': ['Sports'],
        'Travel': ['Travel']
    }
    allowed_usages = occasion_mapping.get(occasion, ['Casual'])

    filtered = fashion_data[
        (fashion_data['gender'].str.lower() == gender.lower()) &
        (fashion_data['basecolour'].isin(recommended_colors)) &
        (fashion_data['usage'].isin(allowed_usages))
    ]

    topwear = filtered[filtered['subcategory'].str.contains('topwear', case=False, na=False)]
    bottomwear = filtered[filtered['subcategory'].str.contains('bottomwear', case=False, na=False)]
    footwear = filtered[filtered['mastercategory'].str.contains('footwear', case=False, na=False)]

    outfit_combinations = []

    if not topwear.empty and not bottomwear.empty and not footwear.empty:
        sampled_topwear = topwear.sample(min(5, len(topwear)))

        for top in sampled_topwear.itertuples():
            harmonious_colors = color_wheel.get(top.basecolour, recommended_colors)
            matching_bottomwear = bottomwear[bottomwear['basecolour'].isin(harmonious_colors)]

            if matching_bottomwear.empty:
                matching_bottomwear = bottomwear

            for bottom in matching_bottomwear.sample(min(2, len(matching_bottomwear))).itertuples():
                matching_footwear = footwear.sample(min(2, len(footwear)))
                for foot in matching_footwear.itertuples():
                    outfit_combinations.append({
                        'Topwear': top.productdisplayname,
                        'Bottomwear': bottom.productdisplayname,
                        'Footwear': foot.productdisplayname
                    })
    return outfit_combinations

# -------------------------------------------
# Sidebar Inputs
# -------------------------------------------

st.sidebar.header("Upload your Details")

uploaded_file = st.sidebar.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

gender = st.sidebar.selectbox("Select Gender", ["Men", "Women"])
occasion = st.sidebar.selectbox("Select Occasion", ["Casual", "Formal", "Party", "Ethnic", "Sports", "Travel"])

# -------------------------------------------
# Main Logic
# -------------------------------------------

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.image(img, channels="BGR", caption="Uploaded Photo", width=300)

    # Detect forehead
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clearer image.")
    else:
        for (x, y, w, h) in faces:
            forehead = img[y:y + int(h * 0.25), x + int(w * 0.3):x + int(w * 0.7)]
            break

        forehead_rgb = cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB)
        avg_rgb = forehead_rgb.mean(axis=(0,1))

        detected_skin_tone = closest_skin_tone(avg_rgb)

        st.success(f"Detected Skin Tone: {detected_skin_tone}")

        # Recommend Outfits
        outfits = recommend_outfits(gender, occasion, detected_skin_tone)

        if outfits:
            st.header("Recommended Outfits and Accessories")

            for idx, outfit in enumerate(outfits[:5]):
                st.subheader(f"Outfit {idx + 1}")
                st.write(f"👚 Topwear: {outfit['Topwear']}")
                st.write(f"👖 Bottomwear: {outfit['Bottomwear']}")
                st.write(f"👟 Footwear: {outfit['Footwear']}")

                # Display outfit images
                for part in ['Topwear', 'Bottomwear', 'Footwear']:
                    for keyword, img_path in outfit_images.items():
                        if keyword in outfit[part].lower() and os.path.exists(img_path):
                            st.image(img_path, width=150)

                # Accessories
                accessories = occasion_accessories.get(occasion.capitalize(), [])
                st.write("🎒 Accessories you can pair:")
                for acc in accessories[:3]:
                    st.write(f"- {acc}")

                st.markdown("---")
        else:
            st.warning("No matching outfits found. Please try a different input!")

else:
    st.info("Please upload a face photo to begin.")

