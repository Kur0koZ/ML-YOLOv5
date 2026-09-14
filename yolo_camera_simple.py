import torch
import cv2

# โหลดโมเดล YOLOv5 (pretrained บน COCO dataset)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# กำหนด threshold
model.conf = 0.25  # confidence threshold - กรอง box ที่มั่นใจน้อยกว่านี้ทิ้ง
model.iou = 0.45   # IoU threshold สำหรับ NMS

# เปิดกล้อง (0 = กล้องตัวแรกของเครื่อง)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # อ่านภาพทีละเฟรมจากกล้อง
    if not ret:
        break

    results = model(frame)  # ส่งเฟรมเข้าโมเดลเพื่อตรวจจับวัตถุ

    while True:
        ret, frame = cap.read()  # อ่านภาพทีละเฟรมจากกล้อง
        if not ret:
            break

        results = model(frame)  # ส่งเฟรมเข้าโมเดลเพื่อตรวจจับวัตถุ

        # วาด bounding box ลงบนเฟรม แล้วแสดงผล
        cv2.imshow("YOLOv5 Camera", results.render()[0])

        if cv2.waitKey(1) & 0xFF == ord("q"):  # กด q เพื่อออก
            break

cap.release()
cv2.destroyAllWindows()