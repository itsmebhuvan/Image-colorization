import torch
import cv2
import numpy as np
from model import ColorizationNet

def colorize_image(img_path, model_path="models/colorization_model.pth", device="cuda"):
    model = ColorizationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    bgr = cv2.imread(img_path)
    bgr = cv2.resize(bgr, (128, 128))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0:1] / 255.0

    L_tensor = torch.from_numpy(L.transpose((2,0,1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_ab = model(L_tensor).cpu().squeeze(0).numpy().transpose((1,2,0))

    pred_ab = pred_ab * 128
    L = L * 255
    lab_out = np.concatenate((L, pred_ab), axis=2).astype("float32")
    bgr_out = cv2.cvtColor(lab_out.astype("uint8"), cv2.COLOR_LAB2BGR)

    cv2.imwrite("outputs/colorized.png", bgr_out)
    print("✅ Colorized image saved at outputs/colorized.png")
