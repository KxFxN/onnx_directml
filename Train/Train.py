from ultralytics import YOLO

# โหลด pretrained model
model = YOLO('./Model/yolov8m.pt')

model.export(format='onnx')

# model.to('cuda')

# # # ที่อยู่ของไฟล์ data.yaml ในโฟลเดอร์ Datasets
# path = '../datasets/custom128.yaml'

# # # เทรนโมเดลโดยใช้ datasets ของคุณ
# results = model.train( data=path, epochs=100,imgsz=640)

# # # ทดสอบโมเดลโดยใช้ validation datasets
# results = model.val()
