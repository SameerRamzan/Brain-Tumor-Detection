import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Previous fixes + Button fixes */
        header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
        
        # .stApp {
        #     background: radial-gradient(circle at 20% 20%, #1e293b 0%, #0f172a 100%);
        # }
                
        .stApp {
        background: linear-gradient(-45deg, #0e1117, #1a1c24, #001f3f, #080a0f);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* 1. Target Primary Buttons (Analyze, Delete, Download) - Keep Gradient */
        .stButton>button[kind="primary"], .stDownloadButton>button {
            background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%) !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            transition: 0.3s;
        }

        /* 2. Target Secondary Buttons (Filenames, Pagination) - Make them look like text/links */
        .stButton>button[kind="secondary"] {
            background: transparent !important;
            border: none !important;
            color: #e2e8f0 !important;
            box-shadow: none !important;
        }
        .stButton>button[kind="secondary"]:hover {
            color: #38bdf8 !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }

        /* Create a 'Result Card' class */
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# Implementation tip for your results section:
def display_results(vgg_acc, resnet_acc):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Uploaded Scan")
        # st.image(your_image)

    with col2:
        st.subheader("Analysis")
        # Use a container with a custom class for better styling
        st.markdown(f'''
            <div class="result-card">
                <p style="margin:0; color:#94a3b8;">VGG16 Confidence</p>
                <h2 style="color:#38bdf8;">{vgg_acc}%</h2>
            </div>
            <div class="result-card">
                <p style="margin:0; color:#94a3b8;">ResNet50 Confidence</p>
                <h2 style="color:#38bdf8;">{resnet_acc}%</h2>
            </div>
        ''', unsafe_allow_html=True)
        
        st.download_button("Download Medical Report", data="...", file_name="report.txt")