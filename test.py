import datetime
import io
import operator

import cv2
import onnxruntime
import torch
import torch.onnx
from model import weatherModel
import torchvision.transforms as transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    model = weatherModel.PFLDInference()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pthfile = 'middle/models/weather-best.pth'
    loaded_model = torch.load(pthfile, map_location=device)
    model.load_state_dict(loaded_model['weather_backbone'])
    model = model.to(device)

    batch_size = 1  # 批处理大小
    input_shape = (3, 224, 224)  # 输入数据,改成自己的输入shape

    # #set the model to inference mode
    model.eval()

    x = torch.randn(batch_size, *input_shape, device=device)  # 生成张量

    export_onnx_file = "test-best-origin.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=10,
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"])  # 输出名


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def detectFaces(img, detector):
    boxes = detector.infer(img)
    return boxes


def test():
    class_ = ['cloudy', 'dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm',
              'shine', 'snow', 'sunrise']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_TYPE = device.__str__()
    if DEVICE_TYPE == 'gpu':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # 初始化 加载性别年龄检测模型
    deepsort_session = onnxruntime.InferenceSession('test-best-origin.onnx', providers=providers)
    head = cv2.imread('data/train_data/fogsmog-4087.jpg')
    to_tensor = transforms.ToTensor()
    img = to_tensor(cv2.resize(head, (224, 224))).unsqueeze_(0)
    inputs = {deepsort_session.get_inputs()[0].name: to_numpy(img)}
    gender_pd = deepsort_session.run(None, inputs)
    max_index, max_number = max(enumerate(gender_pd[0][0]), key=operator.itemgetter(1))
    print(class_[max_index])

if __name__ == '__main__':
    test()