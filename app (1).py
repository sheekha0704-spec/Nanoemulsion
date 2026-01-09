import streamlit as st
import joblib
import pandas as pd
import os

# --- 1. SECURE FILE LOADING ---
def get_path(filename):
    base_path = os.path.dirname(__file__)
    return os.path.join(base_path, filename)

@st.cache_resource
def load_assets():
    try:
        return {
            'model': joblib.load(get_path('drug_nano_predictor_v1.pkl')),
            'features': joblib.load(get_path('model_features.pkl')),
            'le_drug': joblib.load(get_path('le_drug.pkl')),
            'le_oil': joblib.load(get_path('le_oil.pkl')),
            'le_surf': joblib.load(get_path('le_surf.pkl')),
            'drugs': joblib.load(get_path('drug_list.pkl')),
            'oils': joblib.load(get_path('oil_list.pkl')),
            'surfs': joblib.load(get_path('surfactant_list.pkl'))
        }
    except Exception as e:
        st.error(f"⚠️ Missing files on GitHub: {e}")
        return None

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Nanoemulsion AI Optimizer", layout="wide")
st.title("🧪 Nanoemulsion Intelligent Optimizer")

assets = load_assets()

if assets:
    # --- 3. DROP-DOWN SELECTIONS (Categorical Factors) ---
    st.subheader("Formulation Components")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        selected_drug = st.selectbox("Select Drug Name", assets['drugs'])
    with col_b:
        selected_oil = st.selectbox("Select Oil Type", assets['oils'])
    with col_c:
        selected_surf = st.selectbox("Select Surfactant Type", assets['surfs'])

    # --- 4. NUMERICAL PARAMETERS ---
    st.subheader("Physicochemical Factors")
    user_inputs = {}

    # Automatically map selections to encoded values for the model
    user_inputs['Drug_Encoded'] = assets['le_drug'].transform([selected_drug])[0]
    user_inputs['Oil_Encoded'] = assets['le_oil'].transform([selected_oil])[0]
    user_inputs['Surf_Encoded'] = assets['le_surf'].transform([selected_surf])[0]

    # Display only the numerical inputs from your CSV
    # (Excludes the encoded IDs so the user doesn't have to type them)
    num_features = [f for f in assets['features'] if '_Encoded' not in f]
    
    cols = st.columns(2)
    for i, feat in enumerate(num_features):
        with cols[i % 2]:
            # Setting default values based on common ranges in your CSV
            user_inputs[feat] = st.number_input(f"Enter {feat}", value=0.0, key=f"num_{i}")

    # --- 5. PREDICTION LOGIC ---
    if st.button("Analyze Stability", use_container_width=True):
        try:
            # Create DataFrame and enforce the exact column order from training
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df[assets['features']] 
            
            # Get prediction
            prob = assets['model'].predict_proba(input_df)[0][1]
            status = "STABLE" if prob > 0.5 else "UNSTABLE"
            
            # Result UI
            color = "#D4EDDA" if status == "STABLE" else "#F8D7DA"
            st.markdown(f'''
                <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; border: 1px solid #ccc;">
                    <h2 style="color:black; margin:0;">Result: {status}</h2>
                    <p style="color:black; font-size:1.2em;">Confidence: {prob*100:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.warning("Please ensure all 9 .pkl files are uploaded to your GitHub repository.")
