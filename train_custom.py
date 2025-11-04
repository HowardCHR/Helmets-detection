from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')  # 预训练模型

model.train(data='data/data.yaml',device = 'mps', epochs=5, imgsz=640, batch=4)