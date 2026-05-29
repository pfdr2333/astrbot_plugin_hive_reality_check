# Hive Reality Check

这是一个 AstrBot 插件，为 LLM 注册 `check_media_authenticity_hive` 工具，用于核查当前消息或引用消息中的媒体真实性。

## 特性

- 只在用户明确询问“真实性 / 是否 AI 生成 / deepfake / 真假 / 伪造 / 合成”等问题时才应调用工具。
- 支持静态图片：`jpg/jpeg/png/webp`。
- 支持视频：`mp4/webm/m4v`，按 `1 FPS` 本地抽帧后逐帧请求 Hive V3。
- 明确拒绝 `GIF` 与 `APNG`。
- 普通用户默认每日 10 次额度，管理员不限，仍受hive限额。
- 自动裁剪低分来源结果，减少回传 token。

## 安装

将本目录作为插件仓库放入 AstrBot 的 `data/plugins/astrbot_plugin_hive_reality_check/` 目录后重载插件。

## 配置

在 AstrBot WebUI 中配置：

- `enabled`
- `hive_secret_key`
- `hive_v3_daily_request_limit`
- `per_user_daily_limit`
- `allow_video`
- `result_min_score`
- `request_timeout_seconds`

## 获取API KEY
前往[Hive API](https://thehive.ai)注册账号并生成PLAYGROUND API KEY，注册送1刀额度，绑卡送50刀