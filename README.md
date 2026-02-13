# DashScopeRouter

将千问强大的多模态模型，将官方的 DashScope API 接口中转为 OpenAI 协议的接口，将强大的多模态模型能力开放给更大的社区生态

目前针对 Open-WebUI 的功能支持处于高优先等级，即针对图像生成和语音合成与识别的模型支持优先开发，其余多模态模型的支持静待后续开发计划

## TODO

- [ ] 图像生成
  - [ ] 文本生成图像
  - [ ] 图像编辑
    - [ ] 图像编辑-千问
- [ ] 视频生成
- [x] 语音合成
- [x] 语音识别
- [ ] 语音翻译

## 支持的接口

| 路由                       | 方法 | 说明                                  |
| -------------------------- | ---- | ------------------------------------- |
| `/v1/audio/transcriptions` | POST | 音频转文字（OpenAI Whisper API 兼容） |
| `/v1/models`               | GET  | 列出可用模型                          |
| `/health`                  | GET  | 健康检查                              |

## 支持的模型

- `qwen3-asr-flash`
- `qwen3-asr-flash-2025-09-08`
- `qwen3-asr-flash-filetrans`
- `qwen3-asr-flash-filetrans-2025-11-17`

## 快速开始

### 环境变量

| 变量                 | 必须 | 说明                                                             |
| -------------------- | ---- | ---------------------------------------------------------------- |
| `DASHSCOPE_API_KEY`  | 是*  | DashScope API Key（也可在请求 Header 中传入）                    |
| `DASHSCOPE_BASE_URL` | 否   | DashScope API 地址，默认 `https://dashscope.aliyuncs.com/api/v1` |
| `DEFAULT_ASR_MODEL`  | 否   | 默认模型，默认 `qwen3-asr-flash`                                 |
| `DEFAULT_TTS_MODEL`  | 否   | 默认模型，默认 `qwen3-tts-flash`                                 |
| `PORT`               | 否   | 监听端口，默认 `8000`                                            |

### 本地运行

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY=sk-xxx
python main.py
```

### Docker

```bash
docker build -t dashscope-router .
docker run -d -p 8000:8000 -e DASHSCOPE_API_KEY=sk-xxx dashscope-router
```

## 使用示例

### cURL

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-xxx" \
  -F file=@audio.mp3 \
  -F model=qwen3-asr-flash \
  -F response_format=json
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="http://localhost:8000/v1",
)

with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="qwen3-asr-flash",
        file=f,
    )

print(result.text)
```
