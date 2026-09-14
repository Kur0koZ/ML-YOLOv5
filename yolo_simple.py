import torch
import cv2

# โหลดโมเดล YOLOv5 (pretrained บน COCO dataset)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# กำหนด threshold
model.conf = 0.25  # confidence threshold - กรอง box ที่มั่นใจน้อยกว่านี้ทิ้ง
model.iou = 0.45   # IoU threshold สำหรับ NMS

# อ่านภาพจากไฟล์ที่หามา (เปลี่ยน path เป็นรูปของตัวเอง)
img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ส่งภาพเข้าโมเดลเพื่อตรวจจับวัตถุ
results = model(img)

# แสดงผล
results.print()  # พิมพ์สรุปจำนวนวัตถุที่เจอแต่ละคลาส
results.show()   # แสดงภาพพร้อม bounding box
results.save()   # บันทึกภาพผลลัพธ์ (จะเซฟไว้ที่ runs/detect/exp)

# เข้าถึงข้อมูลดิบ: xmin, ymin, xmax, ymax, confidence, class, name
df = results.pandas().xyxy[0]
print(df)