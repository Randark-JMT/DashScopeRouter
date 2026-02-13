# 录音文件识别-千问
# 更新时间：2026-02-12 15:59:50
# https://help.aliyun.com/zh/model-studio/qwen-speech-recognition

import base64
import dashscope
import os
import pathlib

# Support models
# qwen3-asr-flash-filetrans, qwen3-asr-flash-filetrans-2025-11-17
# qwen3-asr-flash, qwen3-asr-flash-2025-09-08

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1，若使用美国地域的模型，需将url替换为：https://dashscope-us.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 请替换为实际的音频文件路径
file_path = "welcome.mp3"
# 请替换为实际的音频文件MIME类型
audio_mime_type = "audio/mpeg"

file_path_obj = pathlib.Path(file_path)
if not file_path_obj.exists():
    raise FileNotFoundError(f"音频文件不存在: {file_path}")

base64_str = base64.b64encode(file_path_obj.read_bytes()).decode()
data_uri = f"data:{audio_mime_type};base64,{base64_str}"

messages = [
    {"role": "system", "content": [{"text": ""}]},  # 配置定制化识别的 Context
    {"role": "user", "content": [{"audio": data_uri}]}
]
response = dashscope.MultiModalConversation.call(
    # 新加坡/美国地域和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 若使用美国地域的模型，需在模型后面加上“-us”后缀，例如qwen3-asr-flash-us
    model="qwen3-asr-flash",
    messages=messages,
    result_format="message",
    asr_options={
        # "language": "zh", # 可选，若已知音频的语种，可通过该参数指定待识别语种，以提升识别准确率
        "enable_itn":False
    }
)
print(response)