import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────
# STEP 1: Load pre-trained YOLO model
# ─────────────────────────────────────────
model = YOLO('yolov8n.pt')  # Auto download hoga pehli baar

# ─────────────────────────────────────────
# STEP 2: Set up video input (webcam)
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)  # 0 = webcam

print("Press 'Q' to quit")

# ─────────────────────────────────────────
# STEP 3: Process each frame
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam not found!")
        break

    # ─────────────────────────────────────
    # STEP 4: Detect objects in frame
    # ─────────────────────────────────────
    results = model(frame, verbose=False)

    # ─────────────────────────────────────
    # STEP 5: Draw bounding boxes + labels
    # ─────────────────────────────────────
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Confidence score
            conf = float(box.conf[0])
            # Class label
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw boxQQ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
            # Draw label
            cv2.putText(frame, f'{label} {conf:.2f}',
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 200, 100), 2)

    # ─────────────────────────────────────
    # STEP 6: Display output
    # ─────────────────────────────────────
    cv2.imshow('Object Detection - Zainab Tariq', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 