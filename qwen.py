# import lmdeploy
# import os
# os.environ["LMDEPLOY_USE_MODELSCOPE"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# pipe = lmdeploy.pipeline("Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b")
# response = pipe(["Hi, pls intro yourself", "Shanghai is"])
# print(response)

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
ckpt_path = "Shanghai_AI_Laboratory/internlm-xcomposer2-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
# `torch_dtype=torch.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，导致显存不足
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')

model = model.eval() 
img_path_list = [
    './hash_folders/9_谈华盛顿.mp4/capture_image/00:00:00,000/0.jpg',
    './hash_folders/9_谈华盛顿.mp4/capture_image/00:00:00,000/1.jpg',
    './hash_folders/9_谈华盛顿.mp4/capture_image/00:00:00,000/2.jpg',
    './hash_folders/9_谈华盛顿.mp4/capture_image/00:00:00,000/3.jpg',
    './hash_folders/9_谈华盛顿.mp4/capture_image/00:00:00,000/4.jpg',
    './hash_folders/9_谈华盛顿.mp4/capture_image/00:00:00,000/5.jpg',
]
images = []
for img_path in img_path_list:
    image = Image.open(img_path).convert("RGB")
    image = model.vis_processor(image)
    images.append(image)
image = torch.stack(images)
query = '<ImageHere> <ImageHere> <ImageHere> <ImageHere> <ImageHere> <ImageHere>这一组图片是从一个视频中采样得到的，请根据这些帧详细总结视频细节。注意以json格式输出，且只需要包含"content"字段。'
with torch.cuda.amp.autocast():
    response, history = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
print(response)