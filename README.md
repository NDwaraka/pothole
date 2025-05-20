Model training :
yolo task=detect mode=train model=yolov8n.pt data=./Data/SegmentationData/YoloV8Aug/data.yaml epochs=100 imgsz=640 project = ./runs3

Using Model:

yolo task=detect mode=predict model="runs3/train/weights/best.pt" source=Data/SegmentationData/YoloV8Aug/test/images save_txt=True project = ./runs6
