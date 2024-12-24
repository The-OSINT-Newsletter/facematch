import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from zipfile import ZipFile
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

# Function to filter matching images
def filter_matching_images(reference_encodings, comparison_files, tolerance):
    new_matching_files = []
    new_match_details = []

    for img_file, name in comparison_files:
        encodings = get_face_encodings(img_file)
        for ref_idx, ref_encoding in enumerate(reference_encodings):
            for comp_encoding in encodings:
                distance = face_recognition.face_distance([ref_encoding], comp_encoding)[0]
                if distance < tolerance:
                    match_percentage = (1 - distance) * 100
                    new_matching_files.append((img_file, name))
                    new_match_details.append((name, f"Target {ref_idx+1}", f"{match_percentage:.2f}%"))
    return new_matching_files, new_match_details

# Function to create a ZIP file for matching images
def create_zip_for_matching_images(matching_files):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        for idx, (image, name) in enumerate(matching_files):
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            zip_file.writestr(f"matching_{idx+1}.png", img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit App
def main():
    st.title("Face Recognition OSINT Tool")

    # Initialize session state
    if "reference_images" not in st.session_state:
        st.session_state.reference_images = []
    if "comparison_images" not in st.session_state:
        st.session_state.comparison_images = []
    if "match_details" not in st.session_state:
        st.session_state.match_details = []
    if "matching_files" not in st.session_state:
        st.session_state.matching_files = []
    if "rerun_analysis" not in st.session_state:
        st.session_state.rerun_analysis = True

    # Step 1: Upload Target Faces
    st.header("1. Upload Target Image(s)")
    target_files = st.file_uploader(
        "Upload images with target faces:",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key="target_uploader"
    )
    if target_files and target_files != st.session_state.reference_images:
        st.session_state.reference_images = target_files
        st.session_state.rerun_analysis = True

    # Step 2: Upload Comparison Images or Scrape a Website
    st.header("2. Upload Comparison Image(s) or Scrape a Website")
    comparison_files = st.file_uploader(
        "Option 1: Upload images to compare against target faces:",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
        key="comparison_uploader"
    )
    if comparison_files:
        st.session_state.comparison_images.extend(
            [(Image.open(img_file), img_file.name) for img_file in comparison_files]
        )
        st.session_state.rerun_analysis = True

    url = st.text_input("Option 2: Enter a website URL to scrape images:")
    if url and st.button("Scrape Images"):
        st.info("Scraping images... please wait.")
        scraped_images = scrape_images_from_url(url)
        if scraped_images:
            st.success(f"Added {len(scraped_images)} new images.")
            st.session_state.comparison_images.extend(scraped_images)
            st.session_state.rerun_analysis = True
        else:
            st.warning("No valid images found.")

    # Step 3: Set Matching Sensitivity
    st.header("3. Set Matching Sensitivity")
    tolerance = st.slider(
        "Adjust Match Sensitivity (Lower is Stricter, Default: 0.6):",
        min_value=0.4, max_value=0.7, value=0.6, step=0.01
    )
    if "last_tolerance" not in st.session_state or st.session_state.last_tolerance != tolerance:
        st.session_state.last_tolerance = tolerance
        st.session_state.rerun_analysis = True

    # Run Analysis
    if st.session_state.reference_images and st.session_state.comparison_images and st.session_state.rerun_analysis:
        st.info("Running analysis... please wait.")
        reference_encodings = []

        # Extract face encodings for all reference images
        for img_file in st.session_state.reference_images:
            encodings = get_face_encodings(Image.open(img_file))
            reference_encodings.extend(encodings)

        # Compare against new comparison images
        new_matching_files, new_match_details = filter_matching_images(
            reference_encodings, st.session_state.comparison_images, tolerance
        )

        # Append new matches to existing results
        st.session_state.matching_files.extend(new_matching_files)
        st.session_state.match_details.extend(new_match_details)
        st.session_state.rerun_analysis = False

    # Display Results
    if st.session_state.matching_files:
        st.header("Results & Summary")

        # Display Target Faces
        st.subheader("Target Faces")
        cols = st.columns(5)
        for idx, img_file in enumerate(st.session_state.reference_images):
            image = Image.open(img_file)
            image.thumbnail((150, 150))
            cols[idx % 5].image(image, caption=f"Target {idx+1}")

        # Display Unique Matching Images
        st.subheader("Matching Images")
        unique_matches = {}
        for idx, (img, name) in enumerate(st.session_state.matching_files):
            if name not in unique_matches:  # Avoid duplicates
                unique_matches[name] = img

        cols = st.columns(5)
        for idx, (name, img) in enumerate(unique_matches.items()):
            img.thumbnail((200, 200))
            cols[idx % 5].image(img, caption=f"Match {idx+1}", use_container_width=True)

        # Export Matching Images
        st.subheader("Export Matching Images")
        zip_buffer = create_zip_for_matching_images(st.session_state.matching_files)
        st.download_button(
            label="Download Matching Images",
            data=zip_buffer,
            file_name="matching_images.zip",
            mime="application/zip"
        )

        # Display Match Details
        st.subheader("Match Details")
        for match in st.session_state.match_details:
            st.write(f"**{match[0]}** matched with **{match[1]}** (Confidence: {match[2]})")

if __name__ == "__main__":
    main()
