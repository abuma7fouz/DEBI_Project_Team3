# =============================================
#  By: TEAM 3 in DEPI 
# =============================================

import streamlit as st
from PIL import Image
import io
import random
import os
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT_DIR = "models"
CLASSES_PICKLE = "pickles/disease_classes.pickle"


# ---------- Soft Color Theme ----------
st.markdown("""
<style>
body {
    background-color: #f4f6f8;
}
.header {
    font-size: 38px;
    font-weight: 700;
    color: #2C3E50;
}
.sub {
    font-size: 20px;
    color: #5d6d7e;
}
.result-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
    border-left: 5px solid #3498db;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
}
.result-title {
    font-size: 22px;
    font-weight: 700;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

DISEASE_ADVICE = {

    "Atelectasis": {
        "link": "https://www.mayoclinic.org/diseases-conditions/atelectasis/symptoms-causes/syc-20369684",
        "advice": """
Advice for Atelectasis (Collapsed Lung Area):
- Use an Incentive Spirometer several times per hour to help reopen the collapsed area.
- Stay well hydrated to thin mucus and improve airway clearance.
- Gentle walking or light activity encourages lung expansion.
- Avoid lying flat; elevate your upper body when resting.
- Seek medical care if breathing worsens or if chest pain increases.
"""
    },

    "Consolidation": {
        "link": "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204",
        "advice": """
Advice for Lung Consolidation:
- Usually indicates pneumonia or lung inflammation, so adequate rest is important.
- Drink warm fluids to help loosen mucus.
- Avoid smoking, vaping, and polluted air.
- If fever persists beyond 48 hours, consider doing CBC and CRP blood tests.
- Seek urgent medical attention if breathing becomes difficult.
"""
    },

    "Infiltration": {
        "link": "https://radiopaedia.org/articles/pulmonary-infiltrate",
        "advice": """
Advice for Lung Infiltration:
- Could be due to early infection or inflammation.
- Increase water intake to help thin mucus.
- Use humidifiers or steam inhalation to ease breathing.
- Avoid smoke, fumes, and dry cold air.
- If symptoms worsen, consult a healthcare provider.
"""
    },

    "Pneumothorax": {
        "link": "https://www.mayoclinic.org/diseases-conditions/collapsed-lung/symptoms-causes/syc-20351565",
        "advice": """
Advice for Pneumothorax (Collapsed Lung):
- Sudden chest pain or severe shortness of breath requires emergency care.
- Avoid heavy lifting, exercise, and flying until medically cleared.
- Stop smoking completely to reduce recurrence risk.
- Follow-up chest X-rays are usually required to confirm recovery.
"""
    },

    "Effusion": {
        "link": "https://my.clevelandclinic.org/health/diseases/17373-pleural-effusion",
        "advice": """
Advice for Pleural Effusion (Fluid Around the Lung):
- Identify the underlying cause, such as heart failure or infection.
- Reduce salt intake to limit fluid accumulation.
- Monitor your breathing; worsening symptoms need urgent medical care.
- A thoracentesis procedure may be needed to analyze or drain the fluid.
"""
    },

    "Cardiomegaly": {
        "link": "https://www.mayoclinic.org/diseases-conditions/enlarged-heart/symptoms-causes/syc-20355436",
        "advice": """
Advice for Cardiomegaly (Enlarged Heart):
- Get an echocardiogram to assess heart pumping function.
- Monitor your blood pressure daily.
- Reduce salt intake significantly to lower heart strain.
- Avoid intense physical activities until cardiology consultation.
- Seek medical help if shortness of breath or swelling increases.
"""
    },

    "Emphysema": {
        "link": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/emphysema",
        "advice": """
Advice for Emphysema:
- Quit smoking immediately, as it slows disease progression.
- Use inhalers and bronchodilators as prescribed.
- Practice breathing techniques such as pursed-lip breathing.
- Stay hydrated and avoid extreme weather or polluted air.
- Join pulmonary rehabilitation if recommended.
"""
    },

    "Fibrosis": {
        "link": "https://www.mayoclinic.org/diseases-conditions/pulmonary-fibrosis/symptoms-causes/syc-20353690",
        "advice": """
Advice for Pulmonary Fibrosis:
- Maintain regular follow-ups with a lung specialist.
- Engage in exercise or pulmonary rehabilitation to strengthen breathing.
- Supplemental oxygen may be necessary depending on oxygen levels.
- Get influenza and pneumonia vaccines to prevent infections.
"""
    },

    "Pneumonia": {
        "link": "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204",
        "advice": """
Advice for Pneumonia:
- Drink plenty of warm fluids to help loosen mucus.
- Rest adequately for several days and avoid heavy effort.
- Begin antibiotics only after medical consultation.
- CBC and CRP tests may be needed if fever persists.
- Seek urgent care if breathing difficulty or chest pain worsens.
"""
    },

    "Edema": {
        "link": "https://www.healthline.com/health/pulmonary-edema",
        "advice": """
Advice for Pulmonary Edema (Fluid in the Lungs):
- Severe breathlessness, chest pain, or bluish lips need immediate emergency care.
- Diuretics may help but should only be taken under medical supervision.
        """
    },

    "Mass": {
        "link": "https://www.cancer.org/cancer/lung-cancer.html",
        "advice": """
Advice for a Lung Mass:
- A CT scan is essential to evaluate the mass accurately.
- Not all masses are cancer; many are benign.
- Avoid smoking immediately.
- Follow up with a specialist for biopsy or monitoring as recommended.
"""
    },

    "Nodule": {
        "link": "https://www.lung.org/lung-health-diseases/lung-disease-lookup/lung-nodules",
        "advice": """
Advice for Lung Nodule:
- A CT scan is usually required for proper assessment.
- Many small nodules are benign and only need follow-up.
- Smoking avoidance is strongly recommended.
- Follow your doctor's recommended imaging schedule.
"""
    },

    "Hernia": {
        "link": "https://www.mayoclinic.org/diseases-conditions/hiatal-hernia/symptoms-causes/syc-20373379",
        "advice": """
Advice for Hiatal Hernia:
- Eat smaller meals and avoid lying down after eating.
- Reduce caffeine, spicy foods, and acidic foods.
- Elevate the head of your bed while sleeping.
- Consult gastroenterology if symptoms persist or worsen.
"""
    },

    "No Finding": {
        "link": "https://www.healthline.com/health/chest-x-ray",
        "advice": """
No abnormalities detected on this chest X-ray:
- Maintain a healthy lifestyle including hydration, physical activity, and avoiding smoking.
- If symptoms continue despite a normal X-ray, follow up with your doctor as some issues do not appear on X-ray imaging.
"""
    }
}



# -------- Preprocessing --------
def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# -------- Load Classes --------
@st.cache_resource
def load_classes(pickle_path=CLASSES_PICKLE):
    if not os.path.exists(pickle_path):
        st.error(f"❌ Missing classes file: {pickle_path}")
        return []
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

# -------- Load Model --------
@st.cache_resource
def load_model(ckpt_path, num_classes):
    if ckpt_path is None:
        return None

    base_model = models.resnet50(pretrained=False)
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    base_model.to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model_obj = ckpt["model"]
        if isinstance(model_obj, nn.Module):
            model_obj.to(DEVICE)
            model_obj.eval()
            return model_obj
        elif isinstance(model_obj, dict):
            base_model.load_state_dict(model_obj)
            base_model.eval()
            return base_model

    elif isinstance(ckpt, dict):
        try:
            base_model.load_state_dict(ckpt)
            base_model.eval()
            return base_model
        except:
            st.error("❌ Cannot load checkpoint")
            return None


def preprocess_image_pil(pil_image):
    tf = get_transform()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return tf(pil_image).unsqueeze(0).to(DEVICE)

def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        return torch.sigmoid(logits).cpu().numpy()[0]


# -------- UI --------
st.markdown("<div class='header'> 🩺 Chest X-Ray Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Upload an X-Ray scan and the AI model will identify possible diseases.</div>", unsafe_allow_html=True)
st.markdown("---")


# -------- Sidebar --------
with st.sidebar:
    st.header("⚙ Model Settings")

    uploaded = st.file_uploader("Upload checkpoint (.pth)", type=["pth","pt"])
    if uploaded:
        ckpt_path = "uploaded_ckpt.pth"
        with open(ckpt_path, "wb") as f:
            f.write(uploaded.read())
    else:
        ckpts = [f for f in os.listdir(DEFAULT_CKPT_DIR) if f.endswith(".pth")]
        choice = st.selectbox("Select from models/", ["--none--"] + ckpts)
        ckpt_path = os.path.join(DEFAULT_CKPT_DIR, choice) if choice != "--none--" else None


# -------- Image Upload --------
st.subheader("📌 Upload X-Ray Image")

col1, col2 = st.columns([1,2])

with col1:
    img_file = st.file_uploader("Choose X-Ray image", type=["png","jpg","jpeg"])
    use_sample = st.button("Use Sample Image")

with col2:
    preview = st.empty()

classes = load_classes()
model = load_model(ckpt_path, len(classes)) if ckpt_path else None

img_to_predict = None


# -------- LOAD IMAGE ----------
if use_sample:
    sample_path = os.path.join("sample_xrays", random.choice(os.listdir("sample_xrays")))
    if os.path.exists(sample_path):
        img_to_predict = Image.open(sample_path)
        preview.image(img_to_predict, caption="Sample Image", width=320)   # SMALLER IMAGE
    else:
        st.error("Sample image not found!")

elif img_file:
    img_to_predict = Image.open(io.BytesIO(img_file.read()))
    preview.image(img_to_predict, caption="Uploaded Image", width=320)    # SMALLER IMAGE


# -------- PREDICT ----------
if img_to_predict is not None:

    st.markdown("### 🔍 Prediction Results")

    if model is None:
        st.error("⚠ No model loaded.")
    else:
        tensor = preprocess_image_pil(img_to_predict)
        probs = predict(model, tensor)

        df = pd.DataFrame({"Disease": classes, "Probability": probs})
        df = df.sort_values("Probability", ascending=False)

        st.dataframe(df.style.format({"Probability": "{:.3f}"}), height=350)

        thresh = st.slider("Display diseases above probability:", 0.1, 0.9, 0.5)
        detected = df[df["Probability"] >= thresh]

        st.markdown(f"### 🩺 Detected Diseases (threshold = {thresh:.2f})")

        if detected.empty:
            st.info("No disease detected above threshold.")
        else:
            for _, row in detected.iterrows():
                d = row["Disease"]
                p = row["Probability"]

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{d} — {p:.2f}</div>
                    <p>
                    🔗 <a href="{DISEASE_ADVICE.get(d,{}).get("link","")}" target="_blank">Learn more</a>
                    </p>
                    <p>{DISEASE_ADVICE.get(d,{}).get("advice","No medical advice available.")}</p>
                </div>
                """, unsafe_allow_html=True)


# -------- FOOTER --------
st.markdown("---")
st.caption(f"Running on: **{DEVICE}** — Model Loaded: **{'Yes' if model else 'No'}**")
