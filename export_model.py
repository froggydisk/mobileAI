import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

model = create_mobilenetv2_ssd_lite(2, is_test=True)
model_path = 'models/mb2-ssd-lite-Epoch-99-Loss-2.2699455499649046.pth'
model.load(model_path)

model.eval()

scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter(
    "models/mb2-ssd-lite-Epoch-99-Loss-2.2699455499649046.ptl")

print("model successfully exported")
