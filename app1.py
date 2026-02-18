import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from huggingface_hub import hf_hub_download

from finetune_chexpert import CheXpertClassifier, transform as xray_transform
from finetune_ct import CTModel
from finetune_mri import MRISSLModel

DEVICE = "cpu"

HF_REPO = "RushikeshReddy7/medical-scan-model"

@st.cache_resource
def load_models():

    xray_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="finetuned_chexpert_model.pth"
    )

    ct_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="finetuned_ct_model.pth"
    )

    mri_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="finetuned_mri_model.pth"
    )

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

def generate_xray_report(image):
    image = image.convert("L")
    img_tensor = xray_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = xray_model(img_tensor).squeeze().cpu().numpy()

    abnormality = float(probs.max())

    report = f"""
**Findings Summary:**
- Abnormality likelihood: {"HIGH" if abnormality>0.6 else "MEDIUM"} ({abnormality:.2f})
- Primary suspicious region: Lung
- Suggested conditions:
  - Possible lung abnormality (confidence: {abnormality:.2f})

Model confidence: Medium  
Recommendation: Further clinical correlation advised
"""
    return report


def generate_ct_report(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
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
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(mri_model(img_tensor)).item()

    report = f"""
**Findings Summary:**
- Abnormality likelihood: {"HIGH" if prob>0.6 else "MEDIUM"} ({prob:.2f})
- Primary suspicious region: Brain
- Suggested conditions:
  - Possible lesion (confidence: {prob:.2f})

Model confidence: Medium  
Recommendation: Further clinical correlation advised
"""
    return report


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
