DETAILER_DETECTOR_MODELS = {
    "face_yolo": "bbox/face_yolov8m.pt",
    "hand_yolo": "bbox/hand_yolov8s.pt",
}


def detailer_detector_model(detector: str | None) -> str:
    value = (detector or "face_yolo").strip()
    return DETAILER_DETECTOR_MODELS.get(value, value)
