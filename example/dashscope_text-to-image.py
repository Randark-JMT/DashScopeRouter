# 文本生成图像
# 更新时间：2026-02-10 10:47:38
# https://help.aliyun.com/zh/model-studio/text-to-image
#
# 示例代码以 qwen-image-plus 为例，但同样适用于万相模型。
# SDK 在底层封装了异步处理逻辑，上层接口表现为同步调用（即单次请求并等待最终结果返回）

# Support models
# qwen-image-max, qwen-image-max-2025-12-30, qwen-image-plus, qwen-image-plus-2026-01-09, qwen-image
# wan2.6-t2i, wan2.5-t2i-preview, wan2.2-t2i-plus, wan2.2-t2i-flash, wanx2.1-t2i-plus, wanx2.1-t2i-turbo, wanx2.0-t2i-turbo

from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis
import os
import dashscope

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

prompt = "一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书“义本生知人机同道善思新”，右书“通云赋智乾坤启数高志远”， 横批“智启千问”，字体飘逸，在中间挂着一幅中国风的画作，内容是岳阳楼。"

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
api_key = os.getenv("DASHSCOPE_API_KEY")

print("----同步调用，请等待任务执行----")
rsp = ImageSynthesis.call(
    api_key=api_key,
    model="qwen-image-plus",  # 当前仅qwen-image-plus、qwen-image模型支持异步接口
    prompt=prompt,
    negative_prompt=" ",
    n=1,
    size="1664*928",
    prompt_extend=True,
    watermark=False,
)
print(f"response: {rsp}")
if rsp.status_code == HTTPStatus.OK:
    # 在当前目录下保存图像
    for result in rsp.output.results:
        file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
        with open("./%s" % file_name, "wb+") as f:
            f.write(requests.get(result.url).content)
else:
    print(f"同步调用失败, status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}")

# Response Example
RESPONSE_EXAMPLE = """
{
    "status_code": 200,
    "request_id": "03b1ef03-480d-4ea5-ba52-xxxxxx",
    "code": null,
    "message": "",
    "output": {
        "task_id": "3cefd9bc-fcb2-4de9-a8bc-xxxxxx",
        "task_status": "SUCCEEDED",
        "results": [
            {
                "url": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxxxxx",
                "orig_prompt": "一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书“义本生知人机同道善思新”，右书“通云赋智乾坤启数高志远”， 横批“智启千问”，字体飘逸，在中间挂着一幅中国风的画作，内容是岳阳楼。",
                "actual_prompt": "一副典雅庄重的对联悬挂于中式厅堂正中，整体空间为安静、古色古香的中国传统布置。厅堂内木质家具沉稳大气，墙面为淡色仿古纸张质感，地面铺设深色木质地板，营造出宁静而庄重的氛围。对联以飘逸的书法字体书写，左侧上联为“义本生知人机同道善思新”，右侧下联为“通云赋智乾坤启数高志远”，横批“智启千问”，文字排列对称，墨色深邃，书法流畅有力，体现出浓厚的文化气息与哲思内涵。\n\n对联中央悬挂一幅中国风画作，内容为岳阳楼，楼阁依水而建，背景为浩渺洞庭湖，远处山峦起伏，云雾缭绕，画面采用传统水墨技法绘制，笔触细腻，意境悠远。画作下方为一张中式红木长桌，桌上错落摆放着几件青花瓷器，包括花瓶与茶具，瓷器釉色清透，纹饰典雅，与整体环境风格和谐统一。整体画面风格为中国古典水墨风，空间布局层次分明，氛围宁静雅致，展现出浓厚的东方文化底蕴。"
            }
        ],
        "submit_time": "2025-09-09 13:41:54.041",
        "scheduled_time": "2025-09-09 13:41:54.087",
        "end_time": "2025-09-09 13:42:22.596"
    },
    "usage": {
        "image_count": 1
    }
}
"""