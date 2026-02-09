import pickle
import streamlit as st

st.title("Fish species predictor")
st.write("Predict fish species from length, weight, and length/weight ratio.")

length = st.number_input("Length (cm)", min_value=0.0,
                         max_value=1000.0, value=20.0)
weight = st.number_input("Weight (g)", min_value=0.0,
                         max_value=100000.0, value=200.0)
ratio = st.number_input(
    "Length/Weight ratio (length ÷ weight)",
    min_value=0.0,
    value=(length / weight) if weight != 0 else 0.0,
    help="Provide the precomputed length-to-weight ratio. Default is computed from the current length and weight values.")

input_data = [[length, weight, ratio]]

MODEL_PATH = "decision_fish.pkl"
ENCODER_PATH = "decision_fish_encoder.pkl"

model = None
encoder = None


def load_artifacts():
    global model, encoder
    if model is None:
        try:
            with open(MODEL_PATH, "rb") as mf:
                model = pickle.load(mf)
        except Exception as e:
            st.error(f"Could not load model '{MODEL_PATH}': {e}")
    if encoder is None:
        try:
            with open(ENCODER_PATH, "rb") as ef:
                encoder = pickle.load(ef)
        except Exception as e:
            st.warning(f"Could not load encoder '{ENCODER_PATH}': {e}")


if st.button("Predict species"):
    load_artifacts()
    if model is None:
        st.warning(
            "Model not available. Place the model file 'decision_fish.pkl' in this folder.")
    else:
        with st.spinner("Predicting..."):
            try:
                pred = model.predict(input_data)
                if encoder is not None:
                    try:
                        species = encoder.inverse_transform(pred)
                        st.success(f"Predicted species: {species[0]}")
                    except Exception:
                        st.success(f"Predicted label: {pred[0]}")
                else:
                    st.success(f"Predicted label: {pred[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
