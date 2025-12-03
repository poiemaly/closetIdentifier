import os
import random
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)
BOTTOM_GALLERY_ROOT = "bottom_gallery"  # folder you created earlier


# ===== load trained model + class names =====
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("shirt_model.h5")
    class_names = np.load("class_names.npy", allow_pickle=True).tolist()
    return model, class_names


model, class_names = load_model_and_classes()

# hard-coded recommendations (you can tweak this!)
shirt_to_bottom_mapping = {
    "Blouse": ["Jeans"],
    "Tank": ["Shorts"],
    "Cardigan": ["Dress"],
    "Sweater": ["Skirt"],
    "Hoodie": ["Joggers"],
}


def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_shirt(pil_img):
    x = preprocess_image(pil_img)
    preds = model.predict(x)
    idx = np.argmax(preds, axis=1)[0]
    shirt_category = class_names[idx]
    confidence = float(np.max(preds))
    return shirt_category, confidence


def get_bottom_gallery_images(bottom_label, max_images=3):
    """
    bottom_label: 'jeans', 'shorts', 'skirt', 'joggers', 'dress_pants', etc.
    Returns up to max_images file paths from bottom_gallery/<bottom_label>/
    """
    folder = os.path.join(BOTTOM_GALLERY_ROOT, bottom_label)
    if not os.path.isdir(folder):
        return []

    all_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_files:
        return []

    if len(all_files) <= max_images:
        return all_files
    else:
        return random.sample(all_files, max_images)


# ========== Streamlit UI ==========
st.title("Outfit Generator 👕 + 👖")
st.write("Upload a shirt image. I'll classify it and show example bottoms that pair well with it.")

uploaded_file = st.file_uploader("Choose a shirt image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Your shirt")
        st.image(pil_image, caption="Uploaded image", use_column_width=True)

        if st.button("Classify & suggest outfit"):
            shirt_category, conf = classify_shirt(pil_image)
            st.markdown(f"**Predicted shirt category:** {shirt_category}")
            st.markdown(f"**Confidence:** {conf:.2f}")

            st.session_state["prediction"] = (shirt_category, conf)

    with col_right:
        st.subheader("Suggested bottoms")

        if "prediction" not in st.session_state:
            st.write("Click **'Classify & suggest outfit'** to see recommendations.")
        else:
            shirt_category, conf = st.session_state["prediction"]
            bottom_labels = shirt_to_bottom_mapping.get(
                shirt_category, ["No recommendations available"]
            )

            if bottom_labels == ["No recommendations available"]:
                st.write("No suggestions configured for this shirt type yet.")
            else:
                for bottom_label in bottom_labels[:3]:
                    st.markdown(f"**{bottom_label.replace('_', ' ').title()}**")

                    img_paths = get_bottom_gallery_images(bottom_label, max_images=3)
                    if not img_paths:
                        st.write("_No example images found in bottom_gallery for this type._")
                        continue

                    img_cols = st.columns(len(img_paths))
                    for col, path in zip(img_cols, img_paths):
                        with col:
                            try:
                                img = Image.open(path)
                                st.image(img, use_column_width=True)
                            except Exception:
                                pass
