from ultralytics import YOLO

modelSize = 'x' #options: 'n', 's', 'm', 'l', 'x'

model = YOLO('models/best.pt')

results = model.predict('input_videos/pavelskigoal.mp4', save=True)

print(results[0])