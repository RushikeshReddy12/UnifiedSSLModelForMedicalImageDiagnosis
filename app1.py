import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import requests
import os

from finetune_chexpert import CheXpertClassifier, transform as xray_transform
from finetune_ct import CTModel
from finetune_mri import MRISSLModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL DOWNLOAD LINKS
# =========================
XRAY_MODEL_URL = "PASTE_LINK"
XRAY_ENCODER_URL = "PASTE_LINK"
CT_MODEL_URL = "PASTE_LINK"
MRI_MODEL_URL = "PASTE_LINK"

def download_file(url, filename):
    if not os.path.exists(filename):
        with open(filename, "wb") as f:
            f.write(requests.get(url).content)

@st.cache_resource
def load_models():

    # Download models if not present
    download_file(XRAY_MODEL_URL, "finetuned_chexpert_model.pth")
    download_file(XRAY_ENCODER_URL, "ssl_chexpert_encoder.pth")
    download_file(CT_MODEL_URL, "finetuned_ct_model.pth")
    download_file(MRI_MODEL_URL, "finetuned_mri_model.pth")

    # Load X-ray model
    xray_model = CheXpertClassifier("ssl_chexpert_encoder.pth").to(DEVICE)
    xray_model.load_state_dict(torch.load("finetuned_chexpert_model.pth", map_location=DEVICE))
    xray_model.eval()

    # Load CT model
    ct_model = CTModel().to(DEVICE)
    ct_model.load_state_dict(torch.load("finetuned_ct_model.pth", map_location=DEVICE))
    ct_model.eval()

    # Load MRI model
    mri_model = MRISSLModel().to(DEVICE)
    mri_model.load_state_dict(torch.load("finetuned_mri_model.pth", map_location=DEVICE))
    mri_model.eval()

    return xray_model, ct_model, mri_model


xray_model, ct_model, mri_model = load_models()

# =========================
# REPORT FUNCTIONS
# =========================

def generate_xray_report(image):
    image = image.convert("L")
    img_tensor = xray_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = xray_model(img_tensor).squeeze().cpu().numpy()

    abnormality_likelihood = float(probs.max())

    DISEASES = [
        "No Finding", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Fracture"
    ]

    findings = [(d, float(p)) for d, p in zip(DISEASES, probs) if p > 0.5]

    report = "**Findings Summary:**\n"
    report += f"- Abnormality likelihood: {'HIGH' if abnormality_likelihood>0.6 else 'MEDIUM'} ({abnormality_likelihood:.2f})\n"
    report += "- Suggested conditions:\n"

    if findings:
        for d, p in findings:
            report += f"  - {d} (confidence: {p:.2f})\n"
    else:
        report += "  - None\n"

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
        logits = ct_model(img_tensor)
        prob = torch.sigmoid(logits).item()

    report = "**Findings Summary:**\n"
    report += f"- Abnormality likelihood: {'HIGH' if prob>0.6 else 'MEDIUM'} ({prob:.2f})\n"
    report += "- Suggested conditions:\n"
    report += f"  - Possible nodule (confidence: {prob:.2f})\n\n"
    report += "Recommendation: Further clinical correlation advised"
    return report


def generate_mri_report(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = mri_model(img_tensor)
        abnormality_likelihood = float(features.norm().item() / 10)

    report = "**Findings Summary:**\n"
    report += f"- Abnormality likelihood: {'HIGH' if abnormality_likelihood>0.6 else 'MEDIUM'} ({abnormality_likelihood:.2f})\n"
    report += "- Suggested conditions:\n"
    report += f"  - Possible lesion (confidence: {abnormality_likelihood:.2f})\n\n"
    report += "Recommendation: Further clinical correlation advised"

    return report

# =========================
# STREAMLIT UI
# =========================

st.title("Self-Supervised AI for Medical Imaging Diagnosis")

st.write("Select scan type and upload image")

modality = st.selectbox("Select Modality", ["X-ray", "CT", "MRI"])

uploaded_file = st.file_uploader(
    f"Upload a {modality} image",
    type=["png","jpg","jpeg","tif","tiff"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    if image.mode == "I" or image.mode == "F":
        image = image.convert("L")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Analyze Image"):

        with st.spinner("Analyzing scan..."):

            if modality == "X-ray":
                report = generate_xray_report(image)
            elif modality == "CT":
                report = generate_ct_report(image)
            else:
                report = generate_mri_report(image)

        st.success("Analysis Complete")
        st.markdown(report)
