import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. FILE PATH HELPER ---
def get_path(filename):
    """Ensures the app finds files on GitHub/Streamlit Cloud."""
    return os.path.join(os.path.dirname(__file__), filename)

# --- 2. LOAD ALL 9 ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return {
            'model': joblib.load(get_path('drug_nano_predictor_v1.pkl')),
            'features': joblib.load(get_path('model_features.pkl')),
            'le_drug': joblib.load(get_path('le_drug.pkl')),
            'le_oil': joblib.load(get_path('le_oil.pkl')),
            'le_surf': joblib.load(get_path('le_surf.pkl')),
            'drug_list': joblib.load(get_path('drug_list.pkl')),
            'oil_list': joblib.load(get_path('oil_list.pkl')),
            'surf_list': joblib.load(get_path('surfactant_list.pkl'))
        }
    except Exception as e:
        st.error(f"⚠️ App Error: Missing required files on GitHub. ({e})")
        return None

# --- 3. UI SETUP ---
st.set_page_config(page_title="Nanoemulsion AI Optimizer", layout="wide")
st.title("🧪 Nanoemulsion Intelligent Optimizer")
st.write("Predict stability based on formulation components and physicochemical factors.")

data = load_assets()

if data:
    # --- 4. DROPDOWN MENUS (Categorical Factors) ---
    st.subheader("1. Select Components")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        u_drug = st.selectbox("Drug Name", data['drug_list'])
    with col2:
        u_oil = st.selectbox("Oil Type", data['oil_list'])
    with col3:
        u_surf = st.selectbox("Surfactant Type", data['surf_list'])

    # --- 5. NUMERICAL INPUTS ---
    st.subheader("2. Enter Physicochemical Factors")
    user_inputs = {}

    # Logic: Convert the words from dropdowns into numbers for the model
    user_inputs['Drug_Encoded'] = data['le_drug'].transform([u_drug])[0]
    user_inputs['Oil_Encoded'] = data['le_oil'].transform([u_oil])[0]
    user_inputs['Surf_Encoded'] = data['le_surf'].transform([u_surf])[0]

    # Map remaining numerical columns from your CSV
    num_features = [f for f in data['features'] if '_Encoded' not in f]
    
    c_left, c_right = st.columns(2)
    for i, feat in enumerate(num_features):
        with (c_left if i % 2 == 0 else c_right):
            user_inputs[feat] = st.number_input(f"{feat}", value=0.0)

    # --- 6. PREDICTION ---
    if st.button("Analyze Stability", use_container_width=True):
        try:
            # Arrange data in the exact order the model was trained on
            input_df = pd.DataFrame([user_inputs])[data['features']]
            
            # Predict
            prob = data['model'].predict_proba(input_df)[0][1]
            is_stable = prob > 0.5
            
            # Display Result
            if is_stable:
                st.success(f"✅ RESULT: STABLE ({prob*100:.1f}% Confidence)")
            else:
                st.error(f"❌ RESULT: UNSTABLE ({(1-prob)*100:.1f}% Confidence)")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.info("💡 Please ensure all .pkl files are uploaded to the same folder as app.py on GitHub.")
