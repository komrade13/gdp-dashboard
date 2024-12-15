# Import required libraries
import PIL
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = 'weights/best.pt'

# Setting page layout
st.set_page_config(
    page_title="Retail Object Detection",  # Setting page title
    page_icon="🛒",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Load the YOLO model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Creating sidebar
with st.sidebar:
    st.header("Image Upload")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader("Upload an image here", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    # Model Options
    confidence = float(st.slider("Set Model Confidence", 0, 100, 50)) / 100
    # Overlap Threshold Slider
    overlap_threshold = float(st.slider("Set Overlap Threshold", 0, 100, 50)) / 100

# Creating main page heading
st.title("Retail Object Detection")
st.caption('Upload a photo containing one or multiple retail products, then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Define object classes and prices
products_prices = {
    0: ('Bihun Asam Pedas', 3000),
    1: ('Bihun Laksa', 3000),
    2: ('Charm Non Wing', 4500),
    3: ('Chitato 75g', 3000),
    4: ('Cotton Buds Mode Puf', 5000),
    5: ('Daia Softener Romantic Pink 245g', 5000),
    6: ('Saus Del Monte Extra Hot', 4500),
    7: ('Daimaru Double Foam Tape', 8000),
    8: ('Minyak Goreng Sawit Fitri', 8000),
    9: ('Susu Kental Manis Frisian Flag', 2500),
    10: ('Lem Joyko', 3000),
    11: ('Gula 500gr', 10000),
    12: ('Minyak Kayu Putih 30ml', 6000),
    13: ('Mama Lemon 105ml', 2000),
    14: ('Amplop Merpati', 20000),
    15: ('Mie Sedaap', 3000),
    16: ('Nabati Korean', 2000),
    17: ('Nabati Lava', 2000),
    18: ('Sabun Cair Nuvo Family', 5000),
    19: ('Pepsodent Herbal', 5000),
    20: ('Polytex Sponge', 3000),
    21: ('Pop Ice Rasa Anggur', 1500),
    22: ('Pop Mie Rasa Ayam Bawang', 3500),
    23: ('Royco 8gr', 500),
    24: ('Crackers Saltcheese Khong Guan', 4000),
    25: ('Sambal Terasi Uleg', 3000),
    26: ('Gunting Gunindo OSS', 6500),
    27: ('Indofood Kecap Manis 77g', 2000),
    28: ('Rosina Gula Pasir 200g', 5000),
    29: ('Santan Sun Kara 65ml', 3500),
    30: ('Super Pell 280ml', 5000),
    31: ('Selotip Nachi Tape', 9500),
    32: ('Teh Tjap Angon', 3500),
    33: ('Tepung 500gr', 7000),
    34: ('Tissue Baik', 10000)
}

# Create an empty DataFrame to hold the checkout list
checkout_df = pd.DataFrame(columns=['Product', 'Price per Item', 'Quantity', 'Subtotal'])

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

# Detect objects and calculate quantities and prices
if st.sidebar.button('Detect Objects', key="detect_objects_button"):
    res = model.predict(uploaded_image, conf=confidence, iou=overlap_threshold)
    # Ensure an image is uploaded before proceeding
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        
        # Perform detection
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        detected_items = {}

        # Initialize detected items dictionary
        detected_items = {}

        # Count each detected item by its class ID
        for box in boxes:
            class_id = int(box.cls)  # Convert class to integer for dictionary lookup
            if class_id in detected_items:
                detected_items[class_id] += 1
            else:
                detected_items[class_id] = 1

        # Fill the DataFrame with products, prices, quantities, and total prices
        for class_id, quantity in detected_items.items():
            product_name, price_per_item = products_prices.get(class_id, ('Unknown', 0.0))
            subtotal = price_per_item * quantity
            new_row = pd.DataFrame({
                'Product': [product_name],
                'Price per Item': [price_per_item],
                'Quantity': [quantity],
                'Subtotal': [subtotal]
            })
            checkout_df = pd.concat([checkout_df, new_row], ignore_index=True)

        # Calculate the final price
        final_total = checkout_df['Subtotal'].sum()

        # Display the checkout table in col1
        with col1:
            st.write("### Checkout List")
            st.table(checkout_df)
            st.divider()
            st.header(f"**Grand Total: :blue[Rp. {final_total:,.2f}]**")

        # Display the detected image in col2
        with col2:
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_container_width=True)
    else:
        st.sidebar.warning("Please upload an image first!")
