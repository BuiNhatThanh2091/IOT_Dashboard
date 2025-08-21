import cv2
import numpy as np
from ultralytics import YOLO

def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def check_no_long_object(frame, wrist, box):
    x1, y1, x2, y2 = map(int, box)
    wrist_x, wrist_y = map(int, wrist)
    roi_size = 80
    roi_x1 = max(x1, wrist_x - roi_size // 2)
    roi_x2 = min(x2, wrist_x + roi_size // 2)
    roi_y1 = max(y1, wrist_y - roi_size // 2)
    roi_y2 = min(y2, wrist_y + roi_size // 2)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return True
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.sum(edges > 0)
    edge_threshold = 300
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)
    has_long_line = any(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) > 40 for [[x1, y1, x2, y2]] in lines) if lines is not None else False
    return not (edge_count > edge_threshold or has_long_line)

# Load model
model = YOLO("yolov8n-pose.pt")

# Load image
image_path = "D:\\nam2hk2\\Vanvatketnoi\\RS-YOLOv8-main\\test6.jpg"
frame = cv2.imread(image_path)
if frame is None:
    print("Không thể đọc ảnh.")
    exit()

# Run pose detection
results = model(frame, conf=0.5)

# Đếm số người giơ tay hợp lệ
count_hand_raised = 0

for r in results:
    boxes = r.boxes.xyxy
    keypoints = r.keypoints.xy

    for i, box in enumerate(boxes):
        person_keypoints = keypoints[i].cpu().numpy()
        left_shoulder = person_keypoints[5]
        right_shoulder = person_keypoints[6]
        left_elbow = person_keypoints[7]
        right_elbow = person_keypoints[8]
        left_wrist = person_keypoints[9]
        right_wrist = person_keypoints[10]
        left_hip = person_keypoints[11]
        right_hip = person_keypoints[12]
        left_knee = person_keypoints[13]
        right_knee = person_keypoints[14]
        left_ankle = person_keypoints[15]
        right_ankle = person_keypoints[16]
        x1, y1, x2, y2 = map(int, box)

        left_ankle_y = left_ankle[1]
        right_ankle_y = right_ankle[1]

        hand_raised = False

        if all(pt[1] > 0 for pt in [left_shoulder, left_elbow, left_wrist]):
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if (45 <= angle < 80) or (100 < angle <= 180):
                if check_no_long_object(frame, left_wrist, box):
                    hand_raised = True

        if all(pt[1] > 0 for pt in [right_shoulder, right_elbow, right_wrist]):
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            if (45 <= angle < 80) or (100 < angle <= 180):
                if check_no_long_object(frame, right_wrist, box):
                    hand_raised = hand_raised or True

        ankle_near_bottom = (left_ankle_y > y2 - 50 or right_ankle_y > y2 - 50) and (left_ankle_y > 0 or right_ankle_y > 0)

        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle) if all(pt[1] > 0 for pt in [left_hip, left_knee, left_ankle]) else 0
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle) if all(pt[1] > 0 for pt in [right_hip, right_knee, right_ankle]) else 0
        valid_left_leg = 135 <= left_leg_angle <= 180
        valid_right_leg = 135 <= right_leg_angle <= 180
        is_pedestrian = ankle_near_bottom and (valid_left_leg or valid_right_leg)

        if hand_raised and is_pedestrian:
            count_hand_raised += 1

# In ra kết quả
print(count_hand_raised)
