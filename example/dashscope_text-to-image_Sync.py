# 文本生成图像
# 更新时间：2026-02-10 10:47:38
# https://help.aliyun.com/zh/model-studio/text-to-image
#
# 示例代码以 qwen-image-plus 为例，但同样适用于万相模型。
# SDK 在底层封装了异步处理逻辑，上层接口表现为同步调用（即单次请求并等待最终结果返回）

# Support models (Sync)
# qwen-image-max, qwen-image-max-2025-12-30, qwen-image-plus, qwen-image-plus-2026-01-09, qwen-image

import json
import os
import dashscope
from dashscope import MultiModalConversation

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

messages = [
    {
        "role": "user",
        "content": [
            {
                "text": "一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书“义本生知人机同道善思新”，右书“通云赋智乾坤启数高志远”， 横批“智启千问”，字体飘逸，在中间挂着一幅中国风的画作，内容是岳阳楼。"
            }
        ],
    }
]

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
api_key = os.getenv("DASHSCOPE_API_KEY")

response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-max",
    messages=messages,
    result_format="message",
    stream=False,
    watermark=False,
    prompt_extend=True,
    negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    size="1664*928",
)

if response.status_code == 200:
    print(json.dumps(response, ensure_ascii=False))
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")


# Response Example
RESPONSE_EXAMPLE = """
{
    "status_code": 200,
    "request_id": "d2d1a8c0-325f-9b9d-8b90-xxxxxx",
    "code": "",
    "message": "",
    "output": {
        "text": null,
        "finish_reason": null,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "image": "https://dashscope-result-wlcb.oss-cn-wulanchabu.aliyuncs.com/xxx.png?Expires=xxx"
                        }
                    ]
                }
            }
        ],
        "task_metric": {
            "TOTAL": 1,
            "FAILED": 0,
            "SUCCEEDED": 1
        }
    },
    "usage": {
        "input_tokens": 0,
        "output_tokens": 0,
        "width": 1328,
        "image_count": 1,
        "height": 1328
    }
}
"""
