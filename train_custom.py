from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='data/data.yaml',
    device='cuda:0',
    epochs=80,
    imgsz=640,        # 若头盔特别小：imgsz=800
    batch=-1,         # 自动最大 batch size（12GB 显存可以跑大概 16~32）
    workers=8,
    optimizer='AdamW',
    lr0=0.002,
    mosaic=1.0,       # 保留数据增强，提高泛化
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    project='runs/train',
    name='helmet_yolov8s',
)