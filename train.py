from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-fasternet.yaml')
    model.train(
        data='dataset/data.yaml',
        cache=False,
        imgsz=640,
        epochs=300,
        batch=32,
        close_mosaic=0,
        workers=8,
        optimizer='SGD',
    )
