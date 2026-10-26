# Hive Reality Check

这是一个 AstrBot 插件，为 LLM 注册 `check_media_authenticity_hive` 工具，用于核查当前消息或引用消息中的媒体真实性。

## 例子
<img width="761" height="675" alt="image" src="https://github.com/user-attachments/assets/06b76580-223f-41a2-a74b-77ddea28ac8f" />


## 特性

- 只在用户明确询问“真实性 / 是否 AI 生成 / deepfake / 真假 / 伪造 / 合成”等问题时才应调用工具（可关闭该文字门控）。
- 支持静态图片：`jpg/jpeg/png/webp`。
- 支持视频：`mp4/webm/m4v`。免费 API 下整段上传检测；付费 API 下均匀抽 **5 帧** 后逐帧请求 Hive V3（每视频至多 5 次付费请求）。
- 支持纯文本核查（≤2048 字符，只走免费通道）。
- 不支持 `GIF` 与 `APNG`。
- 免费 API 优先：无需 Secret Key 即可使用；免费请求失败回退到 Secret Key 付费接口。
- 免费 API 请求默认不计入每日限额（可关闭）。
- 普通用户默认每日 10 次额度（付费请求及关闭豁免后的免费请求计入），管理员不限，仍受 Hive 限额。
- 自动裁剪低分来源结果，减少回传 token。

## 安装

将本目录作为插件仓库放入 AstrBot 的 `data/plugins/astrbot_plugin_hive_reality_check/` 目录后重载插件。

## 配置

在 AstrBot WebUI 中配置：

- `enabled`
- `hive_secret_key`（选填；未配置时仅走免费 API）
- `hive_free_api_enabled`（免费 API 优先，失败回退付费；默认开）
- `free_api_exempt_from_quota`（免费请求不计入限额；默认开）
- `require_authenticity_intent`（文字门控：要求用户输入含真实性关键词才调用；默认开）
- `hive_v3_daily_request_limit`
- `per_user_daily_limit`
- `allow_video`
- `result_min_score`
- `request_timeout_seconds`

## 免费 API

插件内置 Hive AI Detector Chrome 扩展的免费接口客户端：

- 注意免费接口存在限流（实测约每分钟 1 次，连续请求返回 429）。429 会被视为免费失败并自动回退付费接口。
- 免费接口限流较严，有条件的注册下方的API

## 获取API KEY
前往[Hive API](https://thehive.ai)注册账号并生成PLAYGROUND API KEY，注册送1刀额度，绑卡送50刀
