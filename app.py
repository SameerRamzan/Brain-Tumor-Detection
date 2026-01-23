import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import base64
import pydicom
from fpdf import FPDF
from datetime import datetime
import concurrent.futures
import styles  # Import the new styles module
import math
import auth_ui # Import the new auth UI
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration 
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Apply Modern Custom Styles
styles.apply_custom_styles()

# Session Persistence (Restore from URL)
if 'access_token' not in st.session_state and 'access_token' in st.query_params:
    st.session_state['access_token'] = st.query_params['access_token']
    st.session_state['username'] = st.query_params.get('username', 'User')
    st.session_state['is_admin'] = str(st.query_params.get('is_admin', 'false')).lower() == 'true'

# Authentication Gate
if 'access_token' not in st.session_state:
    auth_ui.login_register_page()
    st.stop()

# Define headers for authenticated requests
auth_headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_URL}/predict"

# Helper Functions
def display_dicom_image(file_bytes):
    """Reads and displays a DICOM image, converting it to a displayable format."""
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        img_array = ds.pixel_array
        
        # Normalize 12/16-bit DICOM images for 8-bit display
        if img_array.max() > 0:
            img_array = (img_array / img_array.max()) * 255.0
        img_array = img_array.astype('uint8')
        
        # Convert to PIL Image
        return Image.fromarray(img_array)
    except Exception as e:
        st.error(f"Error processing DICOM file for display: {e}")
        return None

def get_prediction(image_bytes, filename, model_name):
    """Sends image to the FastAPI backend and gets the prediction."""
    try:
        files = {'file': (filename, image_bytes, 'image/jpeg')} # Content-type is flexible
        data = {'model_name': model_name}
        response = requests.post(API_ENDPOINT, files=files, data=data, headers=auth_headers, timeout=30)

        # Handle our custom validation error from the API
        if response.status_code == 422:
            error_detail = response.json().get("detail", "The uploaded image is not a valid MRI scan.")
            st.error(f"Validation Error: {error_detail}")
            return None

        response.raise_for_status()  # Raise an exception for other bad status codes (e.g., 500)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the API at `{API_URL}`. Please ensure the backend is running and the `API_URL` environment variable is set correctly.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None

def apply_custom_threshold(prediction, threshold):
    """Updates the prediction label and risk based on a user-defined threshold."""
    if not prediction:
        return None
    
    confidence = prediction.get('confidence', 0)
    
    if confidence >= threshold:
        prediction['label'] = "YES (Tumor Detected)"
        prediction['risk_level'] = "High"
    elif confidence > 0.30:
        prediction['label'] = "Inconclusive / Needs Review"
        prediction['risk_level'] = "Medium"
    else:
        prediction['label'] = "NO (Healthy)"
        prediction['risk_level'] = "Low"
    return prediction

def create_report_pdf(report_data_list, original_image_bytes, heatmap_bytes_list=None):
    """Generates a PDF report from the analysis data."""
    pdf = FPDF()
    pdf.add_page()
    
    # Attempt to add logo (Top Left)
    try:
        # Replace 'logo.png' with your actual logo file path
        pdf.image("logo.png", x=10, y=8, w=25)
    except (FileNotFoundError, RuntimeError, OSError):
        pass # Continue if logo is missing or fails to load

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Brain Tumor Analysis Report", 0, 1, "C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    pdf.ln(10)

    # Analysis Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Analysis Summary", 0, 1)
    
    for i, report_data in enumerate(report_data_list):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"Result from Model: {report_data.get('model', 'N/A')}", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(40, 7, f"  Diagnosis:", 0, 0)
        pdf.cell(0, 7, report_data.get('label', 'N/A'), 0, 1)
        pdf.cell(40, 7, f"  Confidence:", 0, 0)
        pdf.cell(0, 7, f"{report_data.get('confidence', 0):.2%}", 0, 1)
        pdf.ln(3)

    # Images
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Visual Analysis", 0, 1)

    # Define image properties
    original_img_w = 80
    original_img_h = 80 # Assume square
    heatmap_img_w = 80
    heatmap_img_h = 80 # Assume square

    # Check for page break before drawing block of images
    # Required space: title(5) + original_img_h(80) + margin(10) + title(5) + heatmap_img_h(80)
    required_space = 5 + original_img_h + 10 + 5 + heatmap_img_h
    if pdf.get_y() > pdf.h - required_space:
        pdf.add_page()

    # Original Image Section
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "Original Scan", 0, 1, "C")
    pdf.ln(2)
    y_pos_original = pdf.get_y()
    
    # Display Original Image Centered
    try:
        # Use the first report item to get the filename for checking the type
        if report_data_list and report_data_list[0].get('filename', '').lower().endswith('.dcm'):
            original_img = display_dicom_image(original_image_bytes)
        else:
            original_img = Image.open(io.BytesIO(original_image_bytes))
        
        if original_img.mode == 'RGBA': # pyright: ignore[reportOptionalMemberAccess]
            original_img = original_img.convert('RGB')
        
        with io.BytesIO() as temp_buffer:
            original_img.save(temp_buffer, format="JPEG")
            pdf.image(temp_buffer, x=pdf.w / 2 - original_img_w / 2, y=y_pos_original, w=original_img_w, title="Original Scan", type="JPEG")
    except Exception as e:
        pdf.rect(pdf.w / 2 - original_img_w / 2, y_pos_original, original_img_w, original_img_h, 'D')

    # Heatmaps Section
    y_pos_heatmaps_titles = y_pos_original + original_img_h + 10

    if heatmap_bytes_list:
        x_heatmap1 = 10
        x_heatmap2 = pdf.w - heatmap_img_w - 10 # Align second to the right margin

        # Draw titles
        pdf.set_font("Helvetica", "B", 10)
        if len(heatmap_bytes_list) > 0 and heatmap_bytes_list[0]:
            pdf.set_xy(x_heatmap1, y_pos_heatmaps_titles)
            pdf.cell(heatmap_img_w, 5, f"{report_data_list[0].get('model', 'VGG16')} Heatmap", 0, 0, "C")
        if len(heatmap_bytes_list) > 1 and heatmap_bytes_list[1]:
            pdf.set_xy(x_heatmap2, y_pos_heatmaps_titles)
            pdf.cell(heatmap_img_w, 5, f"{report_data_list[1].get('model', 'ResNet50')} Heatmap", 0, 0, "C")

        y_pos_heatmaps_images = y_pos_heatmaps_titles + 7

        # Draw images
        if len(heatmap_bytes_list) > 0 and heatmap_bytes_list[0]:
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes_list[0]))
            with io.BytesIO() as temp_buffer:
                heatmap_img.save(temp_buffer, format="JPEG")
                pdf.image(temp_buffer, x=x_heatmap1, y=y_pos_heatmaps_images, w=heatmap_img_w, type="JPEG")
        if len(heatmap_bytes_list) > 1 and heatmap_bytes_list[1]:
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes_list[1]))
            with io.BytesIO() as temp_buffer:
                heatmap_img.save(temp_buffer, format="JPEG")
                pdf.image(temp_buffer, x=x_heatmap2, y=y_pos_heatmaps_images, w=heatmap_img_w, type="JPEG")

    return bytes(pdf.output())

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'history_table_key' not in st.session_state:
    st.session_state['history_table_key'] = 0
if 'history_page' not in st.session_state:
    st.session_state['history_page'] = 1
if 'current_page_ids' not in st.session_state:
    st.session_state['current_page_ids'] = []
if 'admin_users_page' not in st.session_state:
    st.session_state['admin_users_page'] = 1

def display_single_model_results(prediction_data, model_name):
    """Helper function to display results for one model."""
    st.subheader(model_name)
    if not prediction_data:
        st.warning("Prediction failed or was not run for this model.")
        return None
    
    risk = prediction_data.get("risk_level")
    label = prediction_data.get("label")
    
    if risk == "High":
        st.error(f"**Diagnosis: {label}**")
    elif risk == "Medium":
        st.warning(f"**Diagnosis: {label}**")
    else:
        st.success(f"**Diagnosis: {label}**")
    
    confidence = prediction_data.get("confidence", 0)
    st.metric(label="Model Confidence", value=f"{confidence:.2%}")
    st.progress(confidence)
    
    heatmap_b64 = prediction_data.get("heatmap_base64")
    if heatmap_b64:
        st.markdown("##### Model Attention Heatmap")
        heatmap_bytes = base64.b64decode(heatmap_b64)
        st.image(Image.open(io.BytesIO(heatmap_bytes)), caption="Red areas show model focus.", use_container_width=True)
        return heatmap_bytes
    return None

def update_select_all():
    """Callback to update all row checkboxes based on the header checkbox."""
    select_all_state = st.session_state.get('history_select_all', False)
    current_ids = st.session_state.get('current_page_ids', [])
    for uid in current_ids:
        st.session_state[f"chk_{uid}"] = select_all_state

# Sidebar Configuration
with st.sidebar:
    st.title("Brain Tumor Detection")
    st.write(f"👤 User: **{st.session_state.get('username', 'Unknown')}**")
    if st.session_state.get('is_admin'):
        st.success("🛡️ Admin Access Active")
    if st.button("Logout", key="logout_btn", type="primary"):
        del st.session_state['access_token']
        if 'is_admin' in st.session_state:
            del st.session_state['is_admin']
        st.query_params.clear()
        st.rerun()
        
    st.info("This app runs predictions using VGG16 and ResNet50. You can select specific models below.")
    
    st.subheader("Model Selection")
    selected_models = st.multiselect(
        "Choose models to run:",
        ["VGG16", "ResNet50"],
        default=["VGG16", "ResNet50"]
    )
    
    st.subheader("Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.70, 
        step=0.05,
        help="Threshold for flagging a result as High Risk."
    )

    st.subheader("Session Stats")
    total_scans = len(st.session_state['history'])
    
    all_confidences = []
    for item in st.session_state['history']:
        for key in ["VGG16 Confidence", "ResNet50 Confidence"]:
            val = item.get(key)
            if val and val not in ["Not Run", "N/A"]:
                try:
                    all_confidences.append(float(val.strip('%')))
                except ValueError:
                    pass

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Total Scans", total_scans)
    stat_col2.metric("Avg Conf.", f"{avg_confidence:.1f}%")
    
    st.markdown('<hr style="margin-top: 0; margin-bottom: 10px; border-color: rgba(255, 255, 255, 0.1);">', unsafe_allow_html=True)
    analyze_clicked = st.button("🔬 Analyze the Scan", type="primary", use_container_width=True)
    
# Main Layout with Tabs
st.title("🧠 Brain Tumor Detection System")

# Define base tabs and conditionally insert Admin tab
tabs_list = ["🔍 Analyze", "📜 History", "📊 Dashboard", "👤 Profile", "ℹ️ About"]
if st.session_state.get('is_admin'):
    tabs_list.insert(2, "👑 Admin")

# Create tabs and a map for easy access
all_tabs = st.tabs(tabs_list)
tab_map = {name: tab for name, tab in zip(tabs_list, all_tabs)}


with tab_map["🔍 Analyze"]:
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an MRI Scan",
        type=["jpg", "jpeg", "png", "dcm"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        
        # Use columns for a side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
            h1, h2 = st.columns([2, 1], vertical_alignment="bottom")
            with h1:
                st.header("Uploaded Scan")
            with h2:
                zoom_size = st.slider("Zoom", 200, 800, 600, 50, key="analyze_zoom")
            
            # Display the appropriate image type
            if uploaded_file.type in ["image/jpeg", "image/png"]:
                st.image(Image.open(io.BytesIO(image_bytes)), caption="Original Scan", width=zoom_size)
            else: # Assumes DICOM for other types
                original_image = display_dicom_image(image_bytes)
                if original_image:
                    st.image(original_image, caption="Original DICOM Scan", width=zoom_size)
        
        if analyze_clicked:
            if not selected_models:
                st.warning("Please select at least one model from the sidebar.")
                st.stop()

            vgg_pred = None
            resnet_pred = None
            with st.spinner(f"Running analysis with {', '.join(selected_models)}..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_vgg = executor.submit(get_prediction, image_bytes, uploaded_file.name, "VGG16") if "VGG16" in selected_models else None
                    future_resnet = executor.submit(get_prediction, image_bytes, uploaded_file.name, "ResNet50") if "ResNet50" in selected_models else None
                    
                    if future_vgg: vgg_pred = future_vgg.result()
                    if future_resnet: resnet_pred = future_resnet.result()
            
            # Apply user-defined threshold to predictions
            if vgg_pred: apply_custom_threshold(vgg_pred, confidence_threshold)
            if resnet_pred: apply_custom_threshold(resnet_pred, confidence_threshold)
            
            with col2:
                res_header, res_dl = st.columns([2, 1], vertical_alignment="center")
                with res_header:
                    st.header("🔬 Analysis Results")
                
                vgg_heatmap_bytes = None
                resnet_heatmap_bytes = None

                # Dynamic layout based on selection
                if len(selected_models) == 1:
                    if "VGG16" in selected_models:
                        vgg_heatmap_bytes = display_single_model_results(vgg_pred, "VGG16")
                    elif "ResNet50" in selected_models:
                        resnet_heatmap_bytes = display_single_model_results(resnet_pred, "ResNet50")
                else:
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        if "VGG16" in selected_models:
                            vgg_heatmap_bytes = display_single_model_results(vgg_pred, "VGG16")
                    with res_col2:
                        if "ResNet50" in selected_models:
                            resnet_heatmap_bytes = display_single_model_results(resnet_pred, "ResNet50")
                
                # Save combined results to History
                st.session_state['history'].append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": uploaded_file.name,
                    "VGG16 Diagnosis": vgg_pred.get('label') if vgg_pred else 'Not Run',
                    "VGG16 Confidence": f"{vgg_pred.get('confidence', 0):.2%}" if vgg_pred else 'Not Run',
                    "ResNet50 Diagnosis": resnet_pred.get('label') if resnet_pred else 'Not Run',
                    "ResNet50 Confidence": f"{resnet_pred.get('confidence', 0):.2%}" if resnet_pred else 'Not Run',
                })

                # Add a download button for the combined PDF report
                if vgg_pred or resnet_pred:
                    report_list = []
                    heatmaps_for_pdf = []
                    
                    if vgg_pred: 
                        report_list.append({**vgg_pred, "model": "VGG16"})
                        heatmaps_for_pdf.append(vgg_heatmap_bytes)
                    if resnet_pred: 
                        report_list.append({**resnet_pred, "model": "ResNet50"})
                        heatmaps_for_pdf.append(resnet_heatmap_bytes)
                    
                    pdf_bytes = create_report_pdf(
                        report_data_list=report_list,
                        original_image_bytes=image_bytes,
                        heatmap_bytes_list=heatmaps_for_pdf
                    )
                    with res_dl:
                        # st.markdown('<div style="height: 1px;"></div>', unsafe_allow_html=True)
                        st.download_button(
                            label="📄 Download Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"report_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=False
                        )
    else:
        st.info("Please upload an image file to begin.")

with tab_map["📜 History"]:
    col_header, col_search, col_btn = st.columns([5, 4, 1])
    with col_header:
        st.header("📜 Global Analysis History")
    
    with col_search:
        def reset_page():
            st.session_state['history_page'] = 1
            st.session_state['history_select_all'] = False
            
        search_term = st.text_input(
            "Search", 
            placeholder="🔍 Filter by filename...", 
            label_visibility="collapsed",
            key="history_search",
            on_change=reset_page
        ).strip()
        
    with col_btn:
        st.button("🔄 Refresh", help="Fetch latest data from database")
    
    # Pagination Parameters
    page_size = 10
    current_page = st.session_state['history_page']

    try:
        with st.spinner("Fetching data from Database..."):
            params = {
                "page": current_page, 
                "limit": page_size, 
                "search": search_term,
            }
            response = requests.get(f"{API_URL}/history", params=params, headers=auth_headers)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # Handle response structure (dict with data/total)
            if isinstance(result, dict):
                db_history = result.get("data", [])
                total_count = result.get("total", 0)
            else:
                total_count = 0
                db_history = [] # Fallback

            if db_history:
                # Store IDs for Select All functionality
                st.session_state['current_page_ids'] = [item['id'] for item in db_history]

                # Header Row
                h1, h2, h3, h4, h5 = st.columns([0.5, 3, 2, 2, 2], vertical_alignment="center")
                h1.checkbox("", key="history_select_all", on_change=update_select_all, help="Select All")
                
                h2.markdown("**Filename**")
                h3.markdown("**Diagnosis**")
                h4.markdown("**Confidence**")
                h5.markdown("**Time**")
                st.markdown('<hr style="margin: 5px 0; border-color: rgba(255, 255, 255, 0.1);">', unsafe_allow_html=True)
                
                ids_to_delete = []
                
                for item in db_history:
                    c1, c2, c3, c4, c5 = st.columns([0.5, 3, 2, 2, 2])
                    
                    # Checkbox for deletion
                    if c1.checkbox("", key=f"chk_{item['id']}"):
                        ids_to_delete.append(item['id'])
                    
                    # Filename Button (Click to View)
                    if c2.button(item.get('filename', 'Unknown'), key=f"btn_{item['id']}"):
                        st.session_state['view_scan_data'] = item
                        st.rerun()
                        
                    # Data Display
                    c3.write(item.get('label', '-'))
                    c4.write(f"{item.get('confidence', 0):.2%}")
                    
                    # Timestamp formatting
                    ts = item.get('timestamp', '')
                    try:
                        dt = datetime.fromisoformat(ts)
                        c5.write(dt.strftime("%Y-%m-%d %H:%M"))
                    except:
                        c5.write(ts)
                
                # Delete Action
                if ids_to_delete:
                    st.divider()
                    if st.button(f"🗑️ Delete {len(ids_to_delete)} Selected", type="primary"):
                        with st.spinner("Deleting records..."):
                            try:
                                for record_id in ids_to_delete:
                                    res = requests.delete(f"{API_URL}/history/{record_id}", headers=auth_headers)
                                    res.raise_for_status()
                                st.session_state['history_table_key'] += 1
                                st.session_state['history_select_all'] = False
                                st.success("Deleted!")
                                st.rerun()
                            except requests.exceptions.ConnectionError:
                                st.error(f"Connection Error: Could not connect to API at `{API_URL}`.")
                            except requests.exceptions.HTTPError as e:
                                st.error(f"Failed to delete. Server returned: {e.response.status_code}")

                # Image Viewer Section
                if 'view_scan_data' in st.session_state:
                    view_data = st.session_state['view_scan_data']
                    st.divider()
                    
                    # Header and Zoom Control
                    vh1, vh2 = st.columns([3, 1], vertical_alignment="bottom")
                    with vh1:
                        st.subheader(f"Viewing: {view_data.get('filename', 'Unknown')}")
                    with vh2:
                        zoom_size = st.slider("Zoom Level", min_value=200, max_value=1200, value=400, step=50)
                    
                    # Layout logic: Stack if zoom is large, otherwise side-by-side
                    if zoom_size > 600:
                        v_col1 = st.container()
                        v_col2 = st.container()
                    else:
                        v_col1, v_col2 = st.columns(2)
                    
                    # Helper to fetch and show
                    def show_gridfs_image(file_id, caption, col):
                        if file_id:
                            try:
                                res = requests.get(f"{API_URL}/files/{file_id}", headers=auth_headers, stream=True)
                                res.raise_for_status()
                                    # Handle DICOM or Standard Image
                                if view_data.get('filename', '').lower().endswith('.dcm') and "Original" in caption:
                                        img = display_dicom_image(res.content)
                                else:
                                        img = Image.open(io.BytesIO(res.content))
                                    
                                if img:
                                        with col:
                                            st.image(img, caption=caption, width=zoom_size)
                            except requests.exceptions.ConnectionError:
                                st.warning(f"Could not load {caption} (Connection Error).")
                            except requests.exceptions.HTTPError:
                                st.warning(f"Could not load {caption} (File not found or access denied).")
                            except Exception as e:
                                with col:
                                    st.warning(f"Could not load {caption}.")

                    show_gridfs_image(view_data.get('original_file_id'), "Original Scan", v_col1)
                    show_gridfs_image(view_data.get('heatmap_file_id'), "AI Heatmap", v_col2)

                    if st.button("Close Viewer"):
                        del st.session_state['view_scan_data']
                        st.rerun()
                
                # Pagination Controls
                total_pages = math.ceil(total_count / page_size)
                if total_pages > 1:
                    st.divider()
                    
                    def change_page(page):
                        st.session_state['history_page'] = page
                        st.session_state['history_select_all'] = False

                    c1, c2, c3 = st.columns([1, 8, 1])
                    with c1:
                        if current_page > 1:
                            st.button("◀ Previous", on_click=change_page, args=(current_page - 1,))
                    with c2:
                        # Centered layout for page info and jump input
                        sc1, sc2, sc3 = st.columns([3, 2, 3])
                        with sc2:
                            st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'>Page <b>{current_page}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
                            
                            def jump_page():
                                st.session_state['history_page'] = st.session_state['jump_to_input']
                                st.session_state['history_select_all'] = False

                            st.number_input(
                                "Jump to", min_value=1, max_value=total_pages, value=current_page, label_visibility="collapsed", key="jump_to_input", on_change=jump_page
                            )
                    with c3:
                        if current_page < total_pages:
                            st.button("Next ▶", on_click=change_page, args=(current_page + 1,))
            else:
                st.info("No history found in the database.")
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the API at `{API_URL}`. Please ensure the backend is running and the `API_URL` environment variable is set correctly.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Failed to fetch history from backend. Server returned: {e.response.status_code} {e.response.reason}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

with tab_map["📊 Dashboard"]:
    st.header("📊 Personal Analytics Dashboard")
    
    # Fetch all data (limit 1000)
    with st.spinner("Loading analytics data..."):
        try:
            response = requests.get(
                f"{API_URL}/history", 
                params={"limit": 1000}, 
                headers=auth_headers
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            if data:
                    df = pd.DataFrame(data)
                    
                    # Preprocessing
                    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['date'] = df['timestamp'].dt.date
                    
                    # KPI Row
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    
                    total_scans = len(df)
                    high_risk = df[df.get('risk_level') == 'High']
                    high_risk_count = len(high_risk)
                    avg_conf = df['confidence'].mean()
                    
                    # Most used model
                    top_model = df['model_name'].mode()[0] if 'model_name' in df.columns and not df['model_name'].mode().empty else "N/A"

                    kpi1.metric("Total Scans", total_scans)
                    kpi2.metric("Tumor Detected", f"{high_risk_count}", delta=f"{high_risk_count/total_scans:.1%}" if total_scans else None, delta_color="inverse")
                    kpi3.metric("Avg Confidence", f"{avg_conf:.1%}")
                    kpi4.metric("Preferred Model", top_model)
                    
                    st.divider()
                    
                    # Visualizations
                    plt.style.use('dark_background')
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.subheader("Diagnosis Distribution")
                        if 'risk_level' in df.columns:
                            fig1, ax1 = plt.subplots(figsize=(5, 4))
                            risk_counts = df['risk_level'].value_counts()
                            colors = {'Low': '#4ade80', 'Medium': '#facc15', 'High': '#f87171'}
                            plot_colors = [colors.get(x, '#94a3b8') for x in risk_counts.index]
                            
                            risk_counts.plot.pie(autopct='%1.1f%%', ax=ax1, startangle=90, colors=plot_colors, explode=[0.05]*len(risk_counts))
                            ax1.set_ylabel('')
                            fig1.patch.set_alpha(0)
                            st.pyplot(fig1, use_container_width=True)
                            plt.close(fig1)
                        
                    with c2:
                        st.subheader("Confidence Distribution")
                        fig2, ax2 = plt.subplots(figsize=(5, 4))
                        sns.histplot(df['confidence'], bins=15, kde=True, ax=ax2, color='#38bdf8')
                        ax2.set_xlim(0, 1)
                        ax2.set_xlabel("Confidence Score")
                        ax2.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                        fig2.patch.set_alpha(0)
                        ax2.patch.set_alpha(0)
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                        
                    st.subheader("Scan Activity Timeline")
                    daily_counts = df.groupby('date').size()
                    st.line_chart(daily_counts)
                    
            else:
                    st.info("No data available. Upload and analyze some MRI scans to see your dashboard!")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to API at `{API_URL}`.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Failed to fetch analytics data. Server returned: {e.response.status_code}")
        except Exception as e:
            st.error(f"An error occurred while loading the dashboard: {e}")

if '👑 Admin' in tab_map:
    with tab_map['👑 Admin']:
        st.header("👑 Admin Dashboard")
        
        with st.spinner("Loading admin statistics..."):
            try:
                response = requests.get(f"{API_URL}/admin/stats", headers=auth_headers)
                response.raise_for_status()
                stats = response.json()
                total_users = stats.get("total_users", 0)
                user_activity = stats.get("user_activity", [])

                st.metric("Total Registered Users", total_users)
                st.divider()

                if user_activity:
                        df_activity = pd.DataFrame(user_activity).set_index("username")
                        
                        # Limit to Top 15 for the chart to prevent overcrowding
                        top_n = 15
                        df_chart = df_activity.head(top_n)
                        
                        st.subheader(f"Top {top_n} Active Users")
                        st.bar_chart(df_chart["scan_count"])
                        
                        with st.expander("View Full Activity Data"):
                            st.dataframe(df_activity, use_container_width=True)
                else:
                        st.info("No user activity has been recorded yet.")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to API at `{API_URL}`.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to load admin stats. Server returned: {e.response.status_code} {e.response.reason}")
            except Exception as e:
                st.error(f"An error occurred connecting to the admin endpoint: {e}")
        
        st.subheader("👥 User Management")
        
        def reset_admin_page():
            st.session_state['admin_users_page'] = 1

        search_query = st.text_input(
            "Search Users", 
            placeholder="🔍 Filter by username...", 
            label_visibility="collapsed",
            key="admin_user_search",
            on_change=reset_admin_page
        )
        
        page_size = 5
        current_page = st.session_state['admin_users_page']

        with st.spinner("Loading users..."):
            try:
                params = {"page": current_page, "limit": page_size, "search": search_query}
                u_res = requests.get(f"{API_URL}/admin/users", params=params, headers=auth_headers)
                u_res.raise_for_status()
                data = u_res.json()
                users_list = data.get("data", [])
                total_users = data.get("total", 0)
                    
                st.caption(f"Showing {len(users_list)} of {total_users} users")

                if not users_list:
                        st.info("No users found.")
                else:
                        for u in users_list:
                            with st.container(border=True):
                                c1, c2, c3 = st.columns([3, 2, 3], vertical_alignment="center")
                                
                                with c1:
                                    st.markdown(f"**{u['username']}**")
                                
                                with c2:
                                    is_admin = u.get('is_admin', False)
                                    if is_admin:
                                        st.markdown("🛡️ Admin")
                                    else:
                                        st.markdown("👤 User")
                                
                                with c3:
                                    b1, b2 = st.columns(2)
                                    with b1:
                                        if is_admin:
                                            if st.button("Demote", key=f"demote_{u['username']}", use_container_width=True):
                                                try:
                                                    res = requests.put(f"{API_URL}/admin/users/{u['username']}/role", json={"is_admin": False}, headers=auth_headers)
                                                    res.raise_for_status()
                                                    st.rerun()
                                                except requests.RequestException as e:
                                                    st.error(f"Failed to demote user: {e}")
                                        else:
                                            if st.button("Promote", key=f"promote_{u['username']}", use_container_width=True):
                                                try:
                                                    res = requests.put(f"{API_URL}/admin/users/{u['username']}/role", json={"is_admin": True}, headers=auth_headers)
                                                    res.raise_for_status()
                                                    st.rerun()
                                                except requests.RequestException as e:
                                                    st.error(f"Failed to promote user: {e}")
                                    with b2:
                                        with st.popover("Delete", use_container_width=True):
                                            st.markdown(f"Delete **{u['username']}**?")
                                            if st.button("Confirm", key=f"conf_del_{u['username']}", type="primary", use_container_width=True):
                                                try:
                                                    res = requests.delete(f"{API_URL}/admin/users/{u['username']}", headers=auth_headers)
                                                    res.raise_for_status()
                                                    st.rerun()
                                                except requests.RequestException as e:
                                                    st.error(f"Failed to delete user: {e}")
                        
                        # Pagination Controls
                        total_pages = math.ceil(total_users / page_size)
                        if total_pages > 1:
                            st.divider()
                            c1, c2, c3 = st.columns([1, 8, 1])
                            
                            def change_admin_page(new_page):
                                st.session_state['admin_users_page'] = new_page
                            
                            with c1:
                                if current_page > 1:
                                    st.button("◀ Prev", key="admin_prev", on_click=change_admin_page, args=(current_page - 1,))
                            with c2:
                                st.markdown(f"<div style='text-align: center;'>Page <b>{current_page}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
                            with c3:
                                if current_page < total_pages:
                                    st.button("Next ▶", key="admin_next", on_click=change_admin_page, args=(current_page + 1,))
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to API at `{API_URL}`.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to load users. Server returned: {e.response.status_code} {e.response.reason}")
            except Exception as e:
                st.error(f"An error occurred while loading users: {e}")

with tab_map["👤 Profile"]:
    st.header(f"👤 User: {st.session_state.get('username', 'Unknown')}")
    # st.write(f"**Username:** {st.session_state.get('username', 'Unknown')}")
    
    #add divider with less top margin
    st.markdown('<hr style="margin-top: 0; margin-bottom: 10px; border-color: rgba(255, 255, 255, 0.1);">', unsafe_allow_html=True)
    st.subheader("🔐 Change Password")
    
    with st.form("change_password_form"):
        current_pass = st.text_input("Current Password", type="password")
        new_pass = st.text_input("New Password", type="password")
        confirm_pass = st.text_input("Confirm New Password", type="password")
        
        submit_btn = st.form_submit_button("Update Password", type="primary")
        
        if submit_btn:
            if not current_pass or not new_pass or not confirm_pass:
                st.warning("Please fill in all fields.")
            elif new_pass != confirm_pass:
                st.error("New passwords do not match.")
            else:
                try:
                    res = requests.post(
                        f"{API_URL}/change-password",
                        json={"old_password": current_pass, "new_password": new_pass},
                        headers=auth_headers
                    )
                    if res.status_code == 200:
                        st.success("Password updated successfully!")
                    else:
                        err_msg = res.json().get('detail', 'Failed to update password.')
                        st.error(f"Error: {err_msg}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Connection Error: Could not connect to API at `{API_URL}`.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    st.markdown('<hr style="margin-top: 30px; margin-bottom: 20px; border-color: rgba(255, 50, 50, 0.2);">', unsafe_allow_html=True)
    st.subheader("⚠️ Danger Zone")
    
    with st.expander("Delete Account", expanded=False):
        st.warning("This action is irreversible. All your data, including history and uploaded files, will be permanently deleted.")
        
        with st.form("delete_account_form"):
            del_password = st.text_input("Confirm Password to Delete", type="password")
            del_submit = st.form_submit_button("Permanently Delete Account", type="primary")
            
            if del_submit:
                if not del_password:
                    st.error("Please enter your password to confirm.")
                else:
                    try:
                        res = requests.delete(
                            f"{API_URL}/delete-account",
                            json={"password": del_password},
                            headers=auth_headers
                        )
                        if res.status_code == 200:
                            st.success("Account deleted successfully.")
                            del st.session_state['access_token']
                            st.query_params.clear()
                            st.rerun()
                        else:
                            err_msg = res.json().get('detail', 'Deletion failed.')
                            st.error(f"Error: {err_msg}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Connection Error: Could not connect to API at `{API_URL}`.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: {e}")

with tab_map["ℹ️ About"]:
    st.header("ℹ️ About This Project")
    st.markdown("""
    This Proof of Concept demonstrates a Deep Learning model for detecting brain tumors from MRI scans.""")
    st.header("How It Works")
    st.markdown("""
    1.  **Upload a Scan**: Supports JPG, PNG, and DICOM formats.
    2.  **Select Model**: Choose between VGG16 (Standard) or ResNet50 (Deep).
    3.  **Analyze**: The image is sent to a FastAPI backend for inference.
    4.  **Interpret**: Results include a diagnosis, confidence score, and a Grad-CAM heatmap to show where the model is looking.""")

    st.header("Hi, I’m Muhammad Sameer👋")
    col_img1, col_img2, col_links, _ = st.columns([1, 1, 2, 8])
    with col_img1:
        st.image("https://avatars.githubusercontent.com/u/583231?v=4", width=100)
    with col_img2:
    #     st.image("https://storage.googleapis.com/kaggle-media/bipoc-grant/kaggle-professor-goose.png", width=200)
    # with col_links:
        st.markdown("""
        - [GitHub](https://github.com/SameerRamzan)
        - [LinkedIn](https://linkedin.com/in/muhammad-sameer123/)
        - [Kaggle](https://kaggle.com/SameerRamzan)
        """)
    st.subheader("Data Scientist | ML Engineer | AI Enthusiast")
    st.markdown("""
    I have hands-on experience in building, deploying, and scaling machine learning solutions end to end. My work goes beyond model development—I focus on taking models from experimentation to production-ready systems.
    I specialize in machine learning, deep learning, NLP, and time-series forecasting, with strong expertise in data preprocessing, feature engineering, and model optimization. I use Python as my primary language, working with libraries such as NumPy, Pandas, Scikit-Learn, TensorFlow, and PyTorch.
    On the deployment side, I have experience containerizing models using Docker, developing APIs with FastAPI/Flask, and integrating ML systems into real-world applications. I’m comfortable working across both backend and frontend, enabling me to build complete ML-powered web applications—from model inference to user-facing interfaces.
    I’m passionate about designing scalable, production-ready AI systems that solve meaningful problems and deliver real impact.
    """)