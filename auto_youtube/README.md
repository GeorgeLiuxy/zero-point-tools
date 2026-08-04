# 本地代理控制接口测试

该目录名称保留为 `auto_youtube`，但当前代码并未实现 YouTube 自动化。`main.py` 仅请求本机 `http://localhost:9090/proxies` 并打印代理配置，注释中还保留了选择代理节点的 POST 示例。

## 运行

```bash
pip install requests
cd auto_youtube
python main.py
```

运行前需要在本机启动兼容该接口的代理控制服务，并确认端口、鉴权方式和 API 路径。若控制器需要 Secret，应通过请求头或本地配置传入，避免写入仓库。

## 当前状态

这是一个接口连通性测试脚本，不包含视频搜索、播放、下载、账号操作或 AI 功能。继续开发时建议重命名目录或补充明确的 YouTube 工作流。
