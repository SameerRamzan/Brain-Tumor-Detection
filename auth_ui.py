import streamlit as st
import requests
import styles_auth_ui
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

def login_register_page():

    login_placeholder = st.empty()

    with login_placeholder.container():
        styles_auth_ui.apply_auth_styles()

        # --- Header Section ---
        st.markdown("<h1 style='text-align: center; color: #00D4FF;'>🧠 Brain Tumor Detection AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #008080;'>Secure Diagnostic Access Portal</p>", unsafe_allow_html=True)
        
        # --- Centering the Layout ---
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            tab1, tab2 = st.tabs(["Login", "Create Account"])
            
            # --- LOGIN TAB ---
            with tab1:
                with st.container():
                    st.markdown("### Welcome Back")
                    username = st.text_input("Username", key="login_user", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Log In", type="primary", use_container_width=True):
                        if not username or not password:
                            st.warning("Please enter both credentials.")
                        else:
                            try:
                                response = requests.post(
                                    f"{API_URL}/token", 
                                    data={"username": username, "password": password}
                                )
                                if response.status_code == 200:
                                    token_data = response.json()
                                    st.session_state['access_token'] = token_data['access_token']
                                    st.session_state['username'] = username
                                    st.session_state['is_admin'] = token_data.get('is_admin', False)
                                    st.query_params['access_token'] = token_data['access_token']
                                    st.query_params['username'] = username
                                    st.query_params['is_admin'] = str(token_data.get('is_admin', False)).lower()
                                    login_placeholder.empty()
                                    st.rerun()
                                else:
                                    st.error("Invalid credentials. Please try again.")
                            except Exception as e:
                                st.error(f"Server Connection Error: {e}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("Forgot Password?"):
                        with st.form("reset_password_form"):
                            st.write("Reset your password using your Recovery Key.")
                            rst_user = st.text_input("Username", key="rst_user")
                            rst_key = st.text_input("Recovery Key", type="password", key="rst_key", help="The key you set during registration")
                            rst_new_pass = st.text_input("New Password", type="password", key="rst_new_pass")
                            
                            if st.form_submit_button("Reset Password"):
                                if not rst_user or not rst_key or not rst_new_pass:
                                    st.warning("All fields are required.")
                                else:
                                    try:
                                        resp = requests.post(f"{API_URL}/reset-password", data={
                                            "username": rst_user.strip(),
                                            "recovery_key": rst_key.strip(),
                                            "new_password": rst_new_pass
                                        })
                                        if resp.status_code == 200:
                                            st.success("Password reset! You can now login.")
                                        else:
                                            st.error(resp.json().get("detail", "Reset failed"))
                                    except Exception as e:
                                        st.error(f"Error: {e}")

            # --- REGISTER TAB ---
            with tab2:
                with st.container():
                    st.markdown("### Get Started")
                    new_user = st.text_input("New Username", key="reg_user", placeholder="Choose a unique username")
                    new_pass = st.text_input("New Password", type="password", key="reg_pass", placeholder="Choose a strong password")
                    recovery_key = st.text_input("Recovery Key", key="reg_key", placeholder="Security answer (e.g. Pet's name)", help="Used to reset password if forgotten")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Register Now", type="primary", use_container_width=True):
                        if not new_user or not new_pass or not recovery_key:
                            st.warning("All fields are required.")
                        else:
                            try:
                                response = requests.post(f"{API_URL}/register", data={
                                    "username": new_user, 
                                    "password": new_pass,
                                    "recovery_key": recovery_key
                                })
                                if response.status_code == 200:
                                    st.balloons()
                                    st.success("Account created! You can now login.")
                                else:
                                    error_detail = response.json().get('detail', 'Registration failed')
                                    st.error(error_detail)
                            except Exception as e:
                                st.error(f"Error: {e}")