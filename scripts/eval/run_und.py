import torch
from PIL import Image
import requests
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils

def to_device_tensors(batch, device):
    """只把 batch 里是 Tensor 的项挪到 device；如果是 list of ints 就显式转 tensor。"""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            # 常见是 input_ids / attention_mask 被返回成 list（老版本或分叉）
            # 仅当元素是数字的 list 才安全地转成张量；其他（如像素数据的复杂结构）保持原样
            if len(v) > 0 and (isinstance(v[0], int) or (isinstance(v[0], list) and (len(v[0]) == 0 or isinstance(v[0][0], int)))):
                out[k] = torch.tensor(v, device=device)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},          # ✅ 放入 PIL 图像
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

# 3) 构建 text prompt（不立即 tokenize），并从 messages 中提取视觉输入
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 提取并预处理图像/视频张量（官方推荐做法）
image_inputs, video_inputs = process_vision_info(messages)

# 4) 打包输入；padding=True 让 batch 更稳妥；返回 PyTorch 张量并移到模型设备
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
).to(model.device)

# inputs = to_device_tensors(inputs, device=model.device)

# 5) 生成：只保留“新生成”的 tokens 再解码（避免把提示词也解码出来）
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=256)

# 提示词长度
prompt_len = inputs["input_ids"].shape[1]

# 新生成部分
new_tokens = generated_ids[:, prompt_len:]

out_text = processor.batch_decode(
    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(out_text)