import torch
from super_gradients.training import models
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


def export_to_onnx(model, onnx_model_name):
    export_result = model.export(onnx_model_name)
    return export_result


if __name__ == '__main__':
    nas = models.get("yolo_nas_l", pretrained_weights="coco")
    print(export_to_onnx(nas, "yolo_nas_l.onnx"))

    # q_util = SelectiveQuantizer(
    #     default_quant_modules_calibrator_weights="max",
    #     default_quant_modules_calibrator_inputs="histogram",
    #     default_per_channel_quant_weights=True,
    #     default_learn_amax=False,
    #     verbose=True,
    # )
    # q_util.quantize_module(nas)
    #
    # nas.to(device)
    # nas.eval()
    # nas.predict_webcam()
