import streamlit as st
import cv2
import numpy as np
import os
import requests


def download_weights_if_missing():
    weights_path = "dnn_model/custom.weights"
    if not os.path.exists(weights_path):
        url = "https://huggingface.co/nachu1234/fabric-defect-yolo-weights/resolve/main/custom.weights"
        os.makedirs("dnn_model", exist_ok=True)
        with st.spinner("Downloading model weights (first run only)..."):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(weights_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)


class FabricDefectDetector:
    def __init__(self):
        download_weights_if_missing()

        weights_path = "dnn_model/custom.weights"
        cfg_path = "dnn_model/Custom.cfg"
        names_path = "classes.names"

        # Load class names
        with open(names_path, "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        # Load YOLO model
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255)

        # Allow only stain class
        self.classes_allowed = [self.class_names.index("stain")]

    def detect_defects(self, img):
        defect_boxes = []
        class_ids, scores, boxes = self.model.detect(
            img, confThreshold=0.2, nmsThreshold=0.4
        )
        for class_id, score, box in zip(class_ids, scores, boxes):
            if class_id in self.classes_allowed:
                defect_boxes.append(box)
                class_name = self.class_names[int(class_id)]
                confidence = round(float(score) * 100, 2)
                label = f"{class_name} ({confidence}%)"
                cv2.putText(img, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(img,
                              (box[0], box[1]),
                              (box[0] + box[2], box[1] + box[3]),
                              (0, 0, 255), 2)
        return img, len(defect_boxes)


# ---------------- STREAMLIT UI ---------------- #
st.title("🧵 Fabric Defect Detection System")


# Load model once
@st.cache_resource
def load_detector():
    return FabricDefectDetector()


detector = load_detector()

# Option selector
option = st.radio("Choose Input Method:", ["Upload Image", "Use Webcam"])

# ---------------- UPLOAD OPTION ---------------- #
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Original Image", channels="BGR")
        result_img, defect_count = detector.detect_defects(img)
        st.image(result_img, caption="Detected Defects", channels="BGR")
        st.success(f"Total Defects Detected: {defect_count}")

# ---------------- WEBCAM OPTION ---------------- #
elif option == "Use Webcam":
    camera_image = st.camera_input("Capture Fabric Image")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Captured Image", channels="BGR")
        result_img, defect_count = detector.detect_defects(img)
        st.image(result_img, caption="Detected Defects", channels="BGR")
        st.success(f"Total Defects Detected: {defect_count}")
