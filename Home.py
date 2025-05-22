import streamlit as st

# Set page configuration
st.set_page_config(page_title="Home Page", layout="wide")

# Title and Subheading
st.title("CP_Reach:")
st.subheader("Cyber-Physical Systems Reachability Analysis Tool")

# st.write("Please select a vehicle type below:")

# Display image (URL or local)
image_url = "images/logo.png"
st.image(image_url)

# # Buttons for navigation
# col1, col2 = st.columns(2)
# with col1:
#     if st.button("Quadrotor"):
#         st.switch_page("Quadrotor")
# with col2:
#     if st.button("Rover"):
#         st.switch_page("Rover")