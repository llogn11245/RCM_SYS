import streamlit as st
import sqlite3
import pandas as pd
import os
import sys
import time
import random

# Add SELFRec path
sys.path.append('/home/llogn/workspace/temp/SELFRec')

from SELFRec import SELFRec
from util.conf import ModelConf

def get_hotels_data(conn):
    """Lấy dữ liệu hotel từ database và sắp xếp theo stars giảm dần"""
    query = """
    SELECT hotel_id, name, hotel_link, stars, rating, reviews, address, intro, price, room_type, province
    FROM data_table 
    ORDER BY stars DESC, rating DESC
    """
    return pd.read_sql_query(query, conn)

def get_hotels_by_ids(conn, hotel_ids):
    """Lấy thông tin hotels theo danh sách hotel_id"""
    if not hotel_ids:
        return pd.DataFrame()
    
    placeholders = ','.join(['?' for _ in hotel_ids])
    query = f"""
    SELECT hotel_id, name, hotel_link, stars, rating, reviews, address, intro, price, room_type, province
    FROM data_table 
    WHERE hotel_id IN ({placeholders})
    """
    return pd.read_sql_query(query, conn, params=hotel_ids)

def get_hotel_image(hotel_id, images_folder="./hotel_images", used_images=None):
    """Lấy đường dẫn ảnh cho hotel từ folder ảnh"""
    # Kiểm tra folder tồn tại
    if not os.path.exists(images_folder):
        return None
    
    # Lấy danh sách file ảnh trong folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    image_files = sorted([f for f in os.listdir(images_folder) 
                   if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        return None
    
    # Nếu có danh sách ảnh đã dùng, loại bỏ chúng
    if used_images is not None:
        available_images = [f for f in image_files if f not in used_images]
        if available_images:
            # Chọn ảnh đầu tiên còn available
            selected_image = available_images[0]
            used_images.add(selected_image)
            return os.path.join(images_folder, selected_image)
    
    # Fallback: dùng hash nếu không có tracking
    image_index = hash(str(hotel_id)) % len(image_files)
    selected_image = image_files[image_index]
    
    return os.path.join(images_folder, selected_image)

def display_hotel_card(hotel_row, rank=None, images_folder="./img", used_images=None):
    """Hiển thị thông tin hotel dưới dạng card với ảnh bên trái và thông tin bên phải"""
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Lấy ảnh từ folder
        image_path = get_hotel_image(hotel_row['hotel_id'], images_folder, used_images)
        
        if image_path and os.path.exists(image_path):
            st.image(image_path, width=400, caption=f"Hotel ID: {hotel_row['hotel_id']}")
        else:
            # Fallback nếu không có ảnh
            st.image("https://via.placeholder.com/200x150?text=Hotel+Image", 
                     width=400, caption=f"Hotel ID: {hotel_row['hotel_id']}")
    
    with col2:
        title = f"#{rank} - {hotel_row['name']}" if rank else hotel_row['name']
        st.subheader(title)
        
        st.write(f"**Rating:** {hotel_row['stars']}⭐")
        st.write(f"**Địa chỉ:** {hotel_row['address']}")
        st.write(f"**Tỉnh/Thành phố:** {hotel_row['province']}")
        st.write(f"**Loại phòng:** {hotel_row['room_type']}")
        
        if pd.notna(hotel_row['price']):
            try:
                price_value = float(hotel_row['price'])
                st.write(f"**Giá:** {price_value:,.0f} VND")
            except (ValueError, TypeError):
                st.write(f"**Giá:** {hotel_row['price']}")
        else:
            st.write("**Giá:** Liên hệ")
        
        if pd.notna(hotel_row['hotel_link']):
            st.link_button("Xem chi tiết", hotel_row['hotel_link'])

class RecommenderWrapper:
    """Wrapper để train và inference với SELFRec"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.conf = None
        
    def train(self, progress_callback=None):
        """Train model và trả về model đã train"""
        conf_path = f'/home/llogn/workspace/temp/SELFRec/conf/{self.model_name}.yaml'
        
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"Config file not found: {conf_path}")
        
        self.conf = ModelConf(conf_path)
        selfrec = SELFRec(self.conf)
        
        # Import model dynamically
        model_type = self.conf['model']['type']
        exec(f"from model.{model_type}.{self.model_name} import {self.model_name}")
        
        self.model = eval(f"{self.model_name}(self.conf, selfrec.training_data, selfrec.test_data)")
        
        # Train model
        self.model.print_model_info()
        self.model.build()
        self.model.train()
        
        return self.model
    
    def get_recommendations(self, user_id, top_k=10):
        """Lấy recommendations cho user"""
        if self.model is None:
            raise ValueError("Model chưa được train!")
        
        # Check if user exists
        if not self.model.data.contain_user(user_id):
            return []
        
        # Get predictions
        scores = self.model.predict(user_id)
        
        # Get top-k items
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Convert to item names/ids
        recommendations = []
        for idx in top_indices:
            if idx in self.model.data.id2item:
                item_name = self.model.data.id2item[idx]
                recommendations.append(item_name)
        
        return recommendations

def main():
    st.set_page_config(
        page_title="Hotel Recommendation System Demo",
        page_icon="🏨",
        layout="wide"
    )
    
    st.title("🏨 Hotel Recommendation System Demo")
    st.markdown("---")
    
    # Initialize session state
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    
    # Sidebar - Model Selection & Training
    st.sidebar.header("🤖 Model Configuration")
    
    available_models = [
        'LightGCN', 'SimGCL', 'SGL', 'MF', 'BUIR', 
        'DirectAU', 'MixGCF', 'NCL', 'XSimGCL'
    ]
    
    selected_model = st.sidebar.selectbox(
        "Chọn Model:",
        available_models,
        index=0
    )
    
    # Train button
    if st.sidebar.button("Chọn Model", type="primary"):
        # with st.spinner(f"Đang train model {selected_model}..."):
        try:
            wrapper = RecommenderWrapper(selected_model)
            
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            status_text.text("Initializing...")
            progress_bar.progress(20)
            
            status_text.text("Running...")
            progress_bar.progress(50)
            
            wrapper.train()
            
            progress_bar.progress(100)
            status_text.text("Succeeded!")
            
            st.session_state.trained_model = wrapper
            st.session_state.model_name = selected_model
            
            # st.sidebar.success(f"✅ Model {selected_model} đã train xong!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi train: {str(e)}")
    
    # Show current model status
    if st.session_state.trained_model:
        st.sidebar.info(f"📊 Model hiện tại: {st.session_state.model_name}")
    
    st.sidebar.markdown("---")
    
    # User ID input for recommendations
    st.sidebar.header("👤 User Recommendation")
    user_id = st.sidebar.text_input("Nhập User ID:", placeholder="Ví dụ: u123")
    top_k = st.sidebar.slider("Số lượng recommendations:", 5, 20, 10)
    
    if st.sidebar.button("🔍 Get Recommendations"):
        if not st.session_state.trained_model:
            st.sidebar.warning("⚠️ Vui lòng train model trước!")
        elif not user_id:
            st.sidebar.warning("⚠️ Vui lòng nhập User ID!")
        else:
            with st.spinner("Đang lấy recommendations..."):
                try:
                    recommendations = st.session_state.trained_model.get_recommendations(
                        user_id, top_k
                    )
                    st.session_state.recommendations = recommendations
                    
                    if recommendations:
                        st.sidebar.success(f"✅ Tìm thấy {len(recommendations)} recommendations!")
                    else:
                        st.sidebar.warning("⚠️ User ID không tồn tại trong training data!")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Lỗi: {str(e)}")
    
    # Kết nối database
    conn = sqlite3.connect('./database.db')
    
    try:
        # Main content
        if st.session_state.recommendations:
            st.header(f"🎯 Recommendations cho User: {user_id}")
            st.write(f"Model: **{st.session_state.model_name}**")
            st.markdown("---")
            
            # Lấy thông tin hotels từ recommendations
            hotel_ids = st.session_state.recommendations
            recommended_hotels = get_hotels_by_ids(conn, hotel_ids)
            
            if not recommended_hotels.empty:
                used_images = set()  # Track ảnh đã sử dụng
                for rank, (idx, hotel_row) in enumerate(recommended_hotels.iterrows(), 1):
                    with st.container():
                        display_hotel_card(hotel_row, rank=rank, used_images=used_images)
                        st.markdown("---")
            else:
                st.info("Đang hiển thị danh sách hotel mặc định (recommendations không match với database)")
                # Fallback: hiển thị top hotels
                hotels_df = get_hotels_data(conn)
                used_images = set()  # Track ảnh đã sử dụng
                for idx, hotel_row in hotels_df.head(top_k).iterrows():
                    with st.container():
                        display_hotel_card(hotel_row, used_images=used_images)
                        st.markdown("---")
        else:
            # Default view - show all hotels
            st.header("📋 Danh sách Hotel (Sắp xếp theo rating)")
            st.info("💡 Chọn model, train và nhập User ID ở sidebar để xem recommendations!")
            
            hotels_df = get_hotels_data(conn)
            
            if hotels_df.empty:
                st.warning("Không có dữ liệu hotel trong database.")
            else:
                used_images = set()  # Track ảnh đã sử dụng
                for idx, hotel_row in hotels_df.head(10).iterrows():
                    with st.container():
                        display_hotel_card(hotel_row, used_images=used_images)
                        st.markdown("---")

    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        st.info("Đảm bảo file database.db tồn tại và có bảng phù hợp.")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()