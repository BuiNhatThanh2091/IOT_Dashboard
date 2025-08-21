import cv2
import numpy as np
from ultralytics import YOLO

# Hàm tính góc giữa ba điểm (trả về góc trong khoảng 0-180 độ)
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)  # Vector p2->p1
    v2 = np.array(p3) - np.array(p2)  # Vector p2->p3
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Tránh lỗi ngoài khoảng [-1, 1]
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# Hàm kiểm tra vùng xung quanh cổ tay để phát hiện vật dài
def check_no_long_object(frame, wrist, box):
    x1, y1, x2, y2 = map(int, box)
    wrist_x, wrist_y = map(int, wrist)
    
    # Xác định vùng ROI xung quanh cổ tay (tăng kích thước để phát hiện vật dài tốt hơn)
    roi_size = 80  # Tăng từ 50 lên 80 để bao quát vật dài
    roi_x1 = max(x1, wrist_x - roi_size // 2)
    roi_x2 = min(x2, wrist_x + roi_size // 2)
    roi_y1 = max(y1, wrist_y - roi_size // 2)
    roi_y2 = min(y2, wrist_y + roi_size // 2)
    
    # Cắt vùng ROI từ ảnh gốc
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return True  # Nếu ROI rỗng, giả định không có vật dài
    
    # Chuyển ROI sang ảnh xám và áp dụng ngưỡng Canny để phát hiện cạnh
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Kiểm tra số lượng cạnh trong ROI
    edge_count = np.sum(edges > 0)
    edge_threshold = 300  # Giảm ngưỡng từ 500 xuống 300 để nhạy hơn với vật dài
    
    # Thêm kiểm tra hình dạng: tìm các đường thẳng dài bằng Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)
    has_long_line = False
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 40:  # Nếu có đường thẳng dài hơn 40 pixel, coi là vật dài
                has_long_line = True
                break
    
    # Có vật dài nếu số lượng cạnh cao hoặc có đường thẳng dài
    return not (edge_count > edge_threshold or has_long_line)

# Khởi tạo mô hình YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")  # Thay bằng yolov8s-pose.pt hoặc yolov8m-pose.pt nếu muốn chính xác hơn

# Đường dẫn đến ảnh của bạn
image_path = "D:\\nam2hk2\\Vanvatketnoi\\RS-YOLOv8-main\\test6.jpg"  # Thay bằng đường dẫn thực tế tới ảnh của bạn

# Đọc ảnh
frame = cv2.imread(image_path)
if frame is None:
    print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn!")
    exit()

# Dự đoán với YOLOv8 Pose
results = model(frame, conf=0.5)  # Độ tin cậy tối thiểu là 0.5

# Danh sách kết quả True/False cho từng người trong ảnh
hand_raised_results = []
count = 0
# Duyệt qua các đối tượng được phát hiện
for r in results:
    boxes = r.boxes.xyxy  # Lấy tọa độ bounding box
    keypoints = r.keypoints.xy  # Lấy tọa độ keypoints (dạng tensor)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        person_keypoints = keypoints[i].cpu().numpy()  # Chuyển keypoints về numpy array

        # Lấy tọa độ vai, khuỷu tay, cổ tay, hông, đầu gối và mắt cá chân
        left_shoulder = person_keypoints[5]       # [x, y] của vai trái
        right_shoulder = person_keypoints[6]      # [x, y] của vai phải
        left_elbow = person_keypoints[7]          # [x, y] của khuỷu tay trái
        right_elbow = person_keypoints[8]         # [x, y] của khuỷu tay phải
        left_wrist = person_keypoints[9]          # [x, y] của cổ tay trái
        right_wrist = person_keypoints[10]        # [x, y] của cổ tay phải
        left_hip = person_keypoints[11]           # [x, y] của hông trái
        right_hip = person_keypoints[12]          # [x, y] của hông phải
        left_knee = person_keypoints[13]          # [x, y] của đầu gối trái
        right_knee = person_keypoints[14]         # [x, y] của đầu gối phải
        left_ankle = person_keypoints[15]         # [x, y] của mắt cá chân trái
        right_ankle = person_keypoints[16]        # [x, y] của mắt cá chân phải
        left_ankle_y = left_ankle[1]              # y của mắt cá chân trái
        right_ankle_y = right_ankle[1]            # y của mắt cá chân phải

        # Kiểm tra xem tay có giơ lên và thỏa mãn góc (45–80 hoặc 100–180 độ) không
        hand_raised = False
        left_arm_angle = 0.0
        right_arm_angle = 0.0

        # Tính góc tay trái (vai trái -> khuỷu tay trái -> cổ tay trái)
        if left_shoulder[1] > 0 and left_elbow[1] > 0 and left_wrist[1] > 0:
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Kiểm tra góc tay: phải nằm trong (45–80) hoặc (100–180)
            if (45 <= left_arm_angle < 80) or (100 < left_arm_angle <= 180):
                # Kiểm tra không cầm vật dài ở cổ tay trái
                if check_no_long_object(frame, left_wrist, box):
                    hand_raised = True

        # Tính góc tay phải (vai phải -> khuỷu tay phải -> cổ tay phải)
        if right_shoulder[1] > 0 and right_elbow[1] > 0 and right_wrist[1] > 0:
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            # Kiểm tra góc tay: phải nằm trong (45–80) hoặc (100–180)
            if (45 <= right_arm_angle < 80) or (100 < right_arm_angle <= 180):
                # Kiểm tra không cầm vật dài ở cổ tay phải
                if check_no_long_object(frame, right_wrist, box):
                    hand_raised = hand_raised or True

        # Kiểm tra xem có phải người đi bộ không
        is_pedestrian = False

        # 1. Điều kiện mắt cá chân gần đáy bounding box
        ankle_near_bottom = (left_ankle_y > y2 - 50 or right_ankle_y > y2 - 50) and (left_ankle_y > 0 or right_ankle_y > 0)

        # 2. Tính góc cho chân trái và chân phải (hông -> đầu gối -> mắt cá chân)
        left_leg_angle = 0.0
        right_leg_angle = 0.0

        if left_hip[1] > 0 and left_knee[1] > 0 and left_ankle[1] > 0:
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

        if right_hip[1] > 0 and right_knee[1] > 0 and right_ankle[1] > 0:
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Kiểm tra góc chân có nằm trong khoảng 135–180 độ không
        valid_left_leg = 135 <= left_leg_angle <= 180 if left_leg_angle > 0 else False
        valid_right_leg = 135 <= right_leg_angle <= 180 if right_leg_angle > 0 else False

        # Người đi bộ: thỏa mãn điều kiện mắt cá chân và ít nhất một chân có góc hợp lệ
        is_pedestrian = ankle_near_bottom and (valid_left_leg or valid_right_leg)

        # Xác định kết quả True/False
        result = hand_raised and is_pedestrian
        hand_raised_results.append(result)

        # Hiển thị trực quan để kiểm tra
        label = f"{'True' if result else 'False'} (Arm: {left_arm_angle:.1f}, {right_arm_angle:.1f}) (Leg: {left_leg_angle:.1f}, {right_leg_angle:.1f})"
        color = (0, 255, 0) if result else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Vẽ keypoints và skeleton
    annotated_frame = r.plot()  # Vẽ cả bounding box và keypoints

    # Ghi đè nhãn True/False lên ảnh annotated
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        person_keypoints = keypoints[i].cpu().numpy()
        left_arm_angle = calculate_angle(person_keypoints[5], person_keypoints[7], person_keypoints[9]) if person_keypoints[5][1] > 0 and person_keypoints[7][1] > 0 and person_keypoints[9][1] > 0 else 0.0
        right_arm_angle = calculate_angle(person_keypoints[6], person_keypoints[8], person_keypoints[10]) if person_keypoints[6][1] > 0 and person_keypoints[8][1] > 0 and person_keypoints[10][1] > 0 else 0.0
        left_leg_angle = calculate_angle(person_keypoints[11], person_keypoints[13], person_keypoints[15]) if person_keypoints[11][1] > 0 and person_keypoints[13][1] > 0 and person_keypoints[15][1] > 0 else 0.0
        right_leg_angle = calculate_angle(person_keypoints[12], person_keypoints[14], person_keypoints[16]) if person_keypoints[12][1] > 0 and person_keypoints[14][1] > 0 and person_keypoints[16][1] > 0 else 0.0
        label = f"{'True' if hand_raised_results[i] else 'False'} (Arm: {left_arm_angle:.1f}, {right_arm_angle:.1f}) (Leg: {left_leg_angle:.1f}, {right_leg_angle:.1f})"
        color = (0, 255, 0) if hand_raised_results[i] else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# In kết quả True/False ra console
#print("Kết quả nhận diện người đi bộ giơ tay:", hand_raised_results)

# Hiển thị ảnh kết quả
#cv2.imshow("YOLOv8 Pose - Pedestrian Hand Detection", annotated_frame)
#cv2.waitKey(0)  # Chờ người dùng nhấn phím bất kỳ để đóng
#cv2.destroyAllWindows()

# (Tùy chọn) Lưu ảnh kết quả
#cv2.imwrite("ket_qua_true_false.jpg", annotated_frame)

# Trả về kết quả để xử lý logic tiếp theo (nếu cần)
def process_logic(results):
    count_raising_hand = 0
    for i, result in enumerate(results):
        if result:
            count_raising_hand += 1
            print(f"Người {i+1}: Giơ tay xin đường")
    print(f"Tổng số người giơ tay xin đường: {count_raising_hand}")

process_logic(hand_raised_results)