# Import thư viện YOLO8
from ultralytics import YOLO
from ultralytics.solutions import object_counter

# Import thư viện openCV
import cv2

# Nạp mô hình yolo8 đã được huấn luyên trước
model = YOLO("yolov8n.pt")

# Mở file video chứa đối tượng để xử lý
cap = cv2.VideoCapture("t7.mp4")
assert cap.isOpened(), "Lỗi đọc file"

# Lấy chiều dài, rộng và số frame của video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Định nghĩa vùng chứa các điểm
region_points = [(200, 400), (1000, 400), (1000, 360), (200, 360)]

# Khởi tạo bộ đếm đối tượng
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,reg_pts=region_points,classes_names=model.names,draw_tracks=True)

# Lặp qua các frame để xử lý (đếm)
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Kết thúc video.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)

# Giải phóng biến và cửa sổ ứng dụng
cap.release()
cv2.destroyAllWindows()