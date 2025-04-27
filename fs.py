
import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------
# STEP 1: Load and Clean the Dataset
# -------------------------------------------

# Load dataset
dataset_path = 'style.csv'  # Adjust path
data = pd.read_csv(dataset_path)

# Data Cleaning
critical_columns = ['gender', 'masterCategory', 'baseColour', 'usage']
data_clean = data.dropna(subset=critical_columns)

imputer = SimpleImputer(strategy='most_frequent')
data_clean[['season', 'subCategory', 'articleType']] = imputer.fit_transform(data_clean[['season', 'subCategory', 'articleType']])
data_clean['year'] = data_clean['year'].fillna(method='ffill')
data_clean.columns = [col.lower() for col in data_clean.columns]

# -------------------------------------------
# STEP 2: Face Detection and Skin Tone Estimation
# -------------------------------------------

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Upload your image
image_path = 'photo.jpg'  # <-- Replace with your actual face image
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found. Check your path!")

# Show Original Image
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Uploaded Face Image")
plt.axis('off')
plt.show()

# Detect face
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    raise ValueError("No face detected. Try another photo!")

# Crop forehead
for (x, y, w, h) in faces:
    forehead = img[y:y + int(h * 0.25), x + int(w * 0.3):x + int(w * 0.7)]
    break

# Show Cropped Forehead
plt.figure(figsize=(4,4))
plt.imshow(cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB))
plt.title("Cropped Forehead Region")
plt.axis('off')
plt.show()

# Average RGB
forehead_rgb = cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB)
average_rgb = forehead_rgb.mean(axis=(0,1))

# Skin Tone Mapping
reference_skin_tones = {
    'Deep': np.array([66, 37, 17]),
    'Tan': np.array([157, 122, 84]),
    'Fair': np.array([229, 200, 166]),
    'Pale': np.array([243, 240, 240])
}

def closest_skin_tone(avg_rgb, ref_tones):
    min_dist = float('inf')
    closest = None
    for tone, ref_rgb in ref_tones.items():
        dist = np.linalg.norm(avg_rgb - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest = tone
    return closest

detected_skin_tone = closest_skin_tone(average_rgb, reference_skin_tones)
print(f"\n✅ Detected Skin Tone: {detected_skin_tone}")

# -------------------------------------------
# STEP 3: Color Theory-Based Outfit Recommendation
# -------------------------------------------

# Color Harmony Mapping
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

# Skin Tone → Color Palette Mapping
def get_recommended_colors(skin_tone):
    skin_tone_to_color_palette = {
        'Deep': ['Navy Blue', 'Olive', 'Rust', 'Burgundy', 'Gold'],
        'Tan': ['Coral', 'Sea Green', 'Turquoise Blue', 'Yellow', 'Peach'],
        'Fair': ['Lavender', 'Soft Pink', 'Light Blue', 'Mint Green', 'White'],
        'Pale': ['Off White', 'Silver', 'Pastel Blue', 'Light Grey']
    }
    return skin_tone_to_color_palette.get(skin_tone, [])

# Outfit Recommendation with Color Theory
def recommend_outfits_with_color_theory(gender, occasion, skin_tone, dataset):
    recommended_colors = get_recommended_colors(skin_tone)
    
    occasion_mapping = {
        'Casual': ['Casual', 'Travel', 'Sports'],
        'Formal': ['Formal'],
        'Party': ['Party'],
        'Ethnic': ['Ethnic'],
        'Sports': ['Sports'],
        'Travel': ['Travel']
    }
    allowed_usages = occasion_mapping.get(occasion, ['Casual'])
    
    filtered = dataset[
        (dataset['gender'].str.lower() == gender.lower()) &
        (dataset['basecolour'].isin(recommended_colors)) &
        (dataset['usage'].isin(allowed_usages))
    ]
    
    topwear = filtered[filtered['subcategory'].str.contains('topwear', case=False, na=False)]
    bottomwear = filtered[filtered['subcategory'].str.contains('bottomwear', case=False, na=False)]
    footwear = filtered[filtered['mastercategory'].str.contains('footwear', case=False, na=False)]
    
    outfit_combinations = []
    
    if not topwear.empty and not bottomwear.empty and not footwear.empty:
        sampled_topwear = topwear.sample(min(5, len(topwear)))
        
        for top in sampled_topwear.itertuples():
            top_color = top.basecolour
            harmonious_colors = color_wheel.get(top_color, recommended_colors)
            matching_bottomwear = bottomwear[bottomwear['basecolour'].isin(harmonious_colors)]
            
            if matching_bottomwear.empty:
                matching_bottomwear = bottomwear  # fallback
            
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
# STEP 4: Occasion-Specific Accessory Recommendation
# -------------------------------------------

# Map Occasion to Accessories
occasion_accessories = {
    'Casual': ['Backpack', 'Sunglasses', 'Sneakers', 'Watch'],
    'Formal': ['Leather Belt', 'Wrist Watch', 'Formal Shoes', 'Minimalist Tie'],
    'Party': ['Statement Necklace', 'Clutch Bag', 'Heels', 'Dangly Earrings'],
    'Ethnic': ['Bangles', 'Ethnic Dupatta', 'Kolhapuri Chappals', 'Bindis'],
    'Sports': ['Sports Watch', 'Cap', 'Training Shoes', 'Sweatbands'],
    'Travel': ['Duffel Bag', 'Comfortable Sneakers', 'Crossbody Bag', 'Travel Hat']
}

# -------------------------------------------
# STEP 5: User Input + Display Outfit + Accessories
# -------------------------------------------

print("\n🎯 Now Let's Generate Your Outfit Suggestions!")
gender = input("Enter your Gender (Men/Women): ").strip()
occasion = input("Enter Occasion (Casual/Formal/Party/Ethnic/Sports/Travel): ").strip()

outfits = recommend_outfits_with_color_theory(gender, occasion, detected_skin_tone, data_clean)

if outfits:
    print(f"\n🎉 Outfit Recommendations for {gender} - {occasion} Look (Detected Skin Tone: {detected_skin_tone})\n")
    
    for idx, outfit in enumerate(outfits[:5]):
        print(f"\nOutfit {idx + 1}:")
        print(f"👚 Topwear: {outfit['Topwear']}")
        print(f"👖 Bottomwear: {outfit['Bottomwear']}")
        print(f"👟 Footwear: {outfit['Footwear']}")
        
        occasion_clean = occasion.capitalize()
        accessories = occasion_accessories.get(occasion_clean, ['Minimal Watch', 'Neutral Bag'])
        
        print("🎒 Accessories you can pair:")
        for acc in accessories[:3]:
            print(f" - {acc}")
        
        print("-" * 50)
else:
    print("\n❌ No matching outfits found. Try selecting a different occasion!")


# In[ ]:




