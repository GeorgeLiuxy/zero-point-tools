package main

//
//import (
//	"encoding/csv"
//	"fmt"
//	"github.com/eatmoreapple/openwechat"
//	"os"
//	"regexp"
//)
//
//// 用于匹配手机号的正则表达式（简化版本）
//var phoneRegex = regexp.MustCompile(`\b\d{11}\b`)
//
//func main() {
//	bot := openwechat.DefaultBot(openwechat.Desktop) // 桌面模式
//
//	// 打开文件用于保存提取的手机号 (CSV格式)
//	file, err := os.OpenFile("phone_numbers.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
//	if err != nil {
//		fmt.Println("无法打开文件:", err)
//		return
//	}
//	defer file.Close()
//
//	// 创建一个CSV写入器
//	writer := csv.NewWriter(file)
//	defer writer.Flush()
//
//	// 如果文件是空的，写入CSV表头
//	fileInfo, _ := file.Stat()
//	if fileInfo.Size() == 0 {
//		err = writer.Write([]string{"群组名称", "成员群组昵称", "手机号", "用户原昵称", "来源"})
//		if err != nil {
//			fmt.Println("写入表头时出错:", err)
//			return
//		}
//	}
//
//	// 注册消息处理函数
//	bot.MessageHandler = func(msg *openwechat.Message) {
//		if msg.IsComeFromGroup() {
//			// 获取发送者的昵称
//			sender, _ := msg.Sender()
//			senderInGroup, _ := msg.SenderInGroup()
//
//			// 确保 sender 不为 nil
//			if sender != nil {
//				// 判断群昵称中是否包含手机号
//				if containsPhoneNumber(senderInGroup.DisplayName) {
//					// 提取手机号并保存到文件
//					phones := extractPhoneNumbers(senderInGroup.DisplayName)
//					for _, phone := range phones {
//						savePhoneNumbers(writer, sender.NickName, senderInGroup.DisplayName, phone, senderInGroup.NickName, "群/昵称")
//					}
//				}
//
//				// 判断消息内容中是否包含手机号
//				if containsPhoneNumber(msg.Content) {
//					// 提取手机号并保存到文件
//					phones := extractPhoneNumbers(msg.Content)
//					for _, phone := range phones {
//						savePhoneNumbers(writer, sender.NickName, senderInGroup.DisplayName, phone, senderInGroup.NickName, "群/消息")
//					}
//				}
//			}
//		}
//	}
//
//	// 注册登陆二维码回调
//	bot.UUIDCallback = openwechat.PrintlnQrcodeUrl
//
//	// 登陆
//	if err := bot.Login(); err != nil {
//		fmt.Println(err)
//		return
//	}
//
//	// 阻塞主goroutine, 直到发生异常或者用户主动退出
//	bot.Block()
//}
//
//// 判断字符串中是否包含手机号
//func containsPhoneNumber(text string) bool {
//	return phoneRegex.MatchString(text)
//}
//
//// 提取字符串中的所有手机号
//func extractPhoneNumbers(text string) []string {
//	// 使用正则提取所有手机号
//	matches := phoneRegex.FindAllString(text, -1)
//	return matches
//}
//
//// 保存手机号到CSV文件
//func savePhoneNumbers(writer *csv.Writer, roomName, groupNickName, phone, nickName, source string) {
//	// 写入一行数据
//	//"群组名称", "成员群组昵称", "手机号", "用户原昵称", "来源"
//	record := []string{roomName, groupNickName, phone, nickName, source}
//	err := writer.Write(record)
//	writer.Flush()
//	if err != nil {
//		fmt.Println("保存手机号时出错:", err)
//	} else {
//		fmt.Printf("保存手机号: %s\n", phone)
//	}
//}
