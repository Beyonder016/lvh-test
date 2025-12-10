"""Streamlit app for LVH detection with Grad-CAM visualization."""
from pathlib import Path
from typing import Tuple

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms

# --- Model utilities -------------------------------------------------------
REPO_ID = "Beyonder016/lvh-detector"
WEIGHTS_FILENAME = "model.pt"
LOCAL_WEIGHTS = Path("lvh_model_f1_balanced.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def load_weights(model: torch.nn.Module, weight_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    cleaned_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    return model


@st.cache_resource
def load_model() -> torch.nn.Module:
    model = build_model()
    load_errors: list[str] = []

    if LOCAL_WEIGHTS.exists():
        try:
            return load_weights(model, LOCAL_WEIGHTS)
        except Exception as exc:  # noqa: BLE001 - surface informative message to user
            load_errors.append(f"Local weights failed: {exc}")
            st.warning("Local weights could not be loaded, falling back to Hugging Face Hub.")

    try:
        hub_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILENAME)
        return load_weights(model, Path(hub_path))
    except Exception as exc:  # noqa: BLE001 - propagate failure details to user
        load_errors.append(str(exc))
        joined = "\n".join(load_errors)
        raise RuntimeError(f"Unable to load model weights. Details:\n{joined}") from exc


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise RuntimeError("No convolutional layer found for Grad-CAM.")


def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    target_layer = get_last_conv_layer(model)

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    logits = model(input_tensor)
    prob = torch.sigmoid(logits).squeeze()

    model.zero_grad()
    logits.backward(torch.ones_like(logits))

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1)).squeeze().detach()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    fwd_handle.remove()
    bwd_handle.remove()

    return prob.item(), cam.cpu().numpy()


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
    heatmap_img = Image.fromarray(np.uint8(cm.jet(heatmap)[:, :, :3] * 255)).resize(original.size)
    return Image.blend(original.convert("RGB"), heatmap_img, alpha=0.5)


# --- Streamlit UI ---------------------------------------------------------
st.set_page_config(page_title="LVH Detection via Chest X-Ray", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgba(87, 227, 255, 0.25), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(0, 255, 200, 0.25), transparent 20%),
                        radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.12), transparent 30%),
                        linear-gradient(135deg, #0f172a, #1e3a8a, #0ea5e9);
            color: #e2e8f0;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            border-radius: 18px;
            padding: 1.5rem;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        .accent-text {
            color: #a5f3fc;
        }

        .glass-button button {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.9), rgba(59, 130, 246, 0.9));
            color: #0b1221;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 8px 16px rgba(14, 165, 233, 0.35);
        }
        .glass-button button:hover {
            border-color: rgba(255, 255, 255, 0.8);
            box-shadow: 0 10px 22px rgba(14, 165, 233, 0.55);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            border: 1px dashed rgba(255, 255, 255, 0.35);
            padding: 1rem;
            backdrop-filter: blur(12px);
        }

        .blurred-badge {
            display: inline-block;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.22);
            backdrop-filter: blur(10px);
            color: #e2e8f0;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

hero = st.container()
with hero:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; margin-bottom: 1rem;">
            <div class="blurred-badge">AI Radiology Assistant</div>
            <h1 style="margin: 0.5rem 0;">LVH Detection from Chest X-Ray</h1>
            <p style="margin: 0; font-size: 1.05rem; color: #dbeafe;">
                Upload an X-ray to detect Left Ventricular Hypertrophy and visualize model attention via Grad-CAM.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

model = load_model()

upload_col, info_col = st.columns([1.3, 1])
with upload_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a chest X-ray image", type=["png", "jpg", "jpeg"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Tips for best results</h3>
            <ul style="padding-left: 1.1rem; line-height: 1.5;">
                <li>Use frontal chest X-rays with minimal artifacts.</li>
                <li>Ensure image is clear and not overexposed.</li>
                <li>PNG or JPG formats work best.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded:
    image = Image.open(uploaded)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        probability = torch.sigmoid(model(input_tensor)).item()

    label = "LVH" if probability >= 0.5 else "No LVH"

    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        st.markdown(
            f"""
            <div class="glass-card" style="margin-top: 1rem;">
                <h3 style="margin-top:0;">Prediction</h3>
                <p style="font-size: 1.1rem;">Result: <span class="accent-text"><strong>{label}</strong></span></p>
                <p style="margin-bottom:0;">LVH Confidence: <strong>{probability:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_col2:
        st.markdown(
            "<div class='glass-card glass-button' style='margin-top: 1rem;'>",
            unsafe_allow_html=True,
        )
        generate_clicked = st.button(
            "Generate Grad-CAM Heatmap", use_container_width=True, key="cam_btn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    display_col1, display_col2 = st.columns(2)
    with display_col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if generate_clicked:
        cam_prob, heatmap = generate_gradcam(model, input_tensor)
        overlay = overlay_heatmap(image, heatmap)
        with display_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.image(
                overlay,
                caption=f"Grad-CAM Heatmap (LVH confidence: {cam_prob:.2%})",
                use_column_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
