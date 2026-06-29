from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve()
    .parents[2]
    / "models"
    / "plate_detection.pt"
)


def crop_plate_yolo(img_input):
    model = YOLO(str(MODEL_PATH))
    results = model.predict(img_input, verbose=False)

    result = results[0]

    if len(result.boxes) > 0:
        box = result.boxes[0]

        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

        plate = img_input[int(ymin):int(ymax), int(xmin):int(xmax)]

        return plate, (int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin))
    else:
        return None, None