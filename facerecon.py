import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.parse import urljoin

# Function to scrape images from a website
def scrape_images_from_url(url, min_size=125, max_size=(800, 800)):
    scraped_images = []
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error("Failed to access the website. Please check the URL.")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        img_tags = soup.find_all("img")

        for img_tag in img_tags:
            img_url = img_tag.get("src")
            if not img_url:
                continue

            img_url = urljoin(url, img_url)
            try:
                img_response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(img_response.content)).convert("RGB")
                if img.width > min_size and img.height > min_size:
                    img.thumbnail(max_size)
                    scraped_images.append((img, img_url))
            except Exception:
                continue
    except Exception as e:
        st.error(f"Error scraping website: {e}")
    return scraped_images

# Function to extract face encodings
def get_face_encodings(image_file, scale=0.75, model="cnn"):
    try:
        image = image_file.convert("RGB")
        width, height = image.size
        image = image.resize((int(width * scale), int(height * scale)))
        image_np = np.array(image)
        face_locations = face_recognition.face_locations(image_np, model=model)
        return face_recognition.face_encodings(image_np, face_locations)
    except Exception:
        return []

# Streamlit App
def main():
    st.title("Face Recognition OSINT Tool")

    # Step 1: Upload Target Faces
    st.header("1. Upload Target Image(s)")
    target_files = st.file_uploader(
        "Upload images with target faces:",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"]
    )

    # Step 2: Provide Comparison Images or Scrape a Website
    st.header("2. Provide Comparison Images or Scrape a Website")
    comparison_files = st.file_uploader(
        "Option 1: Upload images to compare against target faces:",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"]
    )
    url = st.text_input("Option 2: Enter a website URL to scrape images:")

    # Step 3: Set Matching Sensitivity
    st.header("3. Set Matching Sensitivity")
    tolerance = st.slider(
        "Adjust Match Sensitivity (Lower is Stricter, Default: 0.6):",
        min_value=0.4, max_value=0.7, value=0.6, step=0.01
    )

    # Step 4: Run Analysis
    if st.button("Run Analysis"):
        st.info("Running analysis... Please wait.")
        matching_results = []

        # Load target images and extract face encodings
        if target_files:
            reference_encodings = []
            for img_file in target_files:
                image = Image.open(img_file)
                encodings = get_face_encodings(image)
                reference_encodings.extend(encodings)

            # Scrape images from URL if provided
            comparison_images = []
            if url:
                st.info("Scraping images from the website...")
                scraped_images = scrape_images_from_url(url)
                comparison_images.extend(scraped_images)

            # Add uploaded comparison images
            if comparison_files:
                for img_file in comparison_files:
                    comparison_images.append((Image.open(img_file), img_file.name))

            # Match faces
            for comp_img, comp_name in comparison_images:
                encodings = get_face_encodings(comp_img)
                for ref_encoding in reference_encodings:
                    for comp_encoding in encodings:
                        distance = face_recognition.face_distance([ref_encoding], comp_encoding)[0]
                        if distance < tolerance:
                            match_percentage = (1 - distance) * 100
                            matching_results.append((comp_name, f"{match_percentage:.2f}% match"))

        # Display results
        if matching_results:
            st.success("Matching completed!")
            for name, match in matching_results:
                st.write(f"Image: {name}, Match: {match}")
        else:
            st.warning("No matches found.")

if __name__ == "__main__":
    main()
