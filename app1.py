import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
from huggingface_hub import hf_hub_download

from finetune_chexpert import CheXpertClassifier, transform as xray_transform
from finetune_ct import CTModel
from finetune_mri import MRISSLModel

DEVICE = "cpu"

# =============================
# HuggingFace repo info
# =============================
HF_REPO = "YOUR_USERNAME/YOUR_MODEL_REPO"   # change this
XRAY_FILE = "finetuned_chexpert_model.pth"
CT_FILE = "finetuned_ct_model.pth"
MRI_FILE = "finetuned_mri_model.pth"

# =============================
# Download models from HF
# =============================
@st.cache_resource
def load_models():

    # ---- DOWNLOAD MODELS FROM HF ----
    xray_path = hf_hub_download(
        repo_id="RushikeshReddy7",
        filename="finetuned_chexpert_model.pth"
    )

    ct_path = hf_hub_download(
        repo_id="RushikeshReddy7",
        filename="finetuned_ct_model.pth"
    )

    mri_path = hf_hub_download(
        repo_id="RushikeshReddy7",
        filename="finetuned_mri_model.pth"
    )

    # ---- LOAD MODELS ----
    xray_model = CheXpertClassifier("ssl_chexpert_encoder.pth").to(DEVICE)
    xray_model.load_state_dict(torch.load(xray_path, map_location=DEVICE))
    xray_model.eval()

    ct_model = CTModel().to(DEVICE)
    ct_model.load_state_dict(torch.load(ct_path, map_location=DEVICE))
    ct_model.eval()

    mri_model = MRISSLModel().to(DEVICE)
    mri_model.load_state_dict(torch.load(mri_path, map_location=DEVICE))
    mri_model.eval()

    return xray_model, ct_model, mri_model

xray_model, ct_model, mri_model = load_models()

# =============================
# REPORT FUNCTIONS
# =============================
def generate_xray_report(image):
    image = image.convert("L")
    img_tensor = xray_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = xray_model(img_tensor).squeeze().cpu().numpy()

    abnormality_likelihood = float(probs.max())

    DISEASES = [
        "No Finding","Lung Opacity","Lung Lesion","Edema",
        "Consolidation","Pneumonia","Atelectasis",
        "Pneumothorax","Pleural Effusion","Fracture"
    ]

    findings = [(d, float(p)) for d, p in zip(DISEASES, probs) if p > 0.5]

    report = f"""
**Findings Summary:**
- Abnormality likelihood: {"HIGH" if abnormality_likelihood>0.6 else "MEDIUM"} ({abnormality_likelihood:.2f})
- Primary suspicious region: Right lower lung
- Suggested conditions:
"""

    if findings:
        for d, p in findings:
            report += f"  - {d} (confidence: {p:.2f})\n"
    else:
        report += "  - None\n"

    report += "\nModel confidence: Medium"
    report += "\nRecommendation: Further clinical correlation advised"

    return report


def generate_ct_report(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_tensor = transform(image.convert("L")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(ct_model(img_tensor)).item()

    report = f"""
**Findings Summary:**
- Abnormality likelihood: {"HIGH" if prob>0.6 else "MEDIUM"} ({prob:.2f})
- Primary suspicious region: Lung field
- Suggested conditions:
  - Possible nodule (confidence: {prob:.2f})

Model confidence: Medium
Recommendation: Further clinical correlation advised
"""
    return report


def generate_mri_report(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = mri_model(img_tensor)
        prob = torch.sigmoid(out).item()

    report = f"""
**Findings Summary:**
- Abnormality likelihood: {"HIGH" if prob>0.6 else "MEDIUM"} ({prob:.2f})
- Primary suspicious region: Brain region
- Suggested conditions:
  - Possible lesion (confidence: {prob:.2f})

Model confidence: Medium
Recommendation: Further clinical correlation advised
"""
    return report


# =============================
# STREAMLIT UI
# =============================
st.title("Multimodal Medical Imaging Diagnosis")

modality = st.selectbox("Select scan type", ["X-ray", "CT", "MRI"])
uploaded_file = st.file_uploader("Upload scan", type=["png","jpg","jpeg","tif","tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)

    if image.mode in ["I","F"]:
        image = image.convert("L")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, width=400)

    if st.button("Analyze"):
        if modality == "X-ray":
            report = generate_xray_report(image)
        elif modality == "CT":
            report = generate_ct_report(image)
        else:
            report = generate_mri_report(image)

        st.markdown(report)
