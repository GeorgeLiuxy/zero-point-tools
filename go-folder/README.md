# 微信群手机号提取 Go 原型

`main.go` 原计划基于 OpenWechat 登录微信，检查群成员昵称或消息中的 11 位手机号，并把结果写入 `phone_numbers.csv`。

## 当前状态

文件中的实现目前全部被注释，目录也没有 `go.mod`，因此不能直接编译或运行。它应被视为设计草稿，而不是可用工具。

## 继续开发

1. 初始化模块并安装 OpenWechat：

   ```bash
   cd go-folder
   go mod init zero-point-tools/wechat-phone-export
   go get github.com/eatmoreapple/openwechat
   ```

2. 审核并取消必要代码的注释。
3. 补充登录失败、重复手机号、CSV 编码和权限处理。
4. 使用测试群验证后再处理真实数据。

手机号和群成员信息属于敏感个人数据，只能在获得成员授权、具备合法目的并符合平台规则的情况下采集和保存。
