#!/usr/bin/env python3
"""Preview the trained XY target on the camera without controlling motors."""
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


WIDTH, HEIGHT = 320, 240


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', default='best_steering_model_xy.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()
    print('Loaded {} on {}. Press Q in the camera window to quit.'.format(
        args.model, device), flush=True)

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        parser.error('Cannot open camera {}'.format(args.camera))

    window = 'Model target preview - no motor control - Q to quit'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                if cv2.waitKey(20) & 0xFF in (ord('q'), ord('Q')):
                    break
                continue
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transforms.functional.resize(image, (224, 224))
            image = transforms.functional.to_tensor(image)
            image = torch.from_numpy(image.numpy()[::-1].copy())
            image = transforms.functional.normalize(
                image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            with torch.no_grad():
                xy = model(image.unsqueeze(0).to(device))[0].cpu().numpy()
            x = float((xy[0] + 1.0) * WIDTH / 2.0)
            y = float((xy[1] + 1.0) * HEIGHT / 2.0)
            preview = frame.copy()
            cv2.line(preview, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT - 1),
                     (255, 255, 0), 1)
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < WIDTH and 0 <= y < HEIGHT:
                cv2.circle(preview, (int(round(x)), int(round(y))), 8,
                           (0, 255, 0), 2)
                label = 'Predicted target: x={:.0f}, y={:.0f}'.format(x, y)
            else:
                label = 'Predicted target is outside image'
            cv2.rectangle(preview, (0, 0), (WIDTH, 25), (0, 0, 0), -1)
            cv2.putText(preview, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
