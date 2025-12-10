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
st.set_page_config(page_title="LVH Detection via Chest X-Ray", layout="centered")

st.title("💓 LVH Detection from Chest X-Ray")
st.markdown(
    "Upload a chest X-ray image to detect Left Ventricular Hypertrophy (LVH) and visualize model attention with Grad-CAM."
)

model = load_model()

uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        probability = torch.sigmoid(model(input_tensor)).item()

    label = "LVH" if probability >= 0.5 else "No LVH"
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence for LVH:** {probability:.2%}")

    if st.button("Generate Grad-CAM Heatmap"):
        cam_prob, heatmap = generate_gradcam(model, input_tensor)
        overlay = overlay_heatmap(image, heatmap)
        st.image(overlay, caption=f"Grad-CAM Heatmap (LVH confidence: {cam_prob:.2%})", use_column_width=True)
