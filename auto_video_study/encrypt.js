// encrypt.js

const crypto = require('crypto');

// PEM 格式的公钥字符串
const publicKeyPem = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvIjlCqMQYjHw1/+A4rT7n8h9y9k5c7EdzqmVyke6R4Cw7qTBh51j6YTQ2pIz0JkNvxgI80ItqeoFeHzyyOScga1uj1xyp0JU7IAoaFkWSeqRXRsaNQrssXEQg6SK/3WEkn1W5ZdVFWGjnsrqpI24JFJt50Nm/vmBMo8bIYRIPvV9yTE4LxDr207ptJO5QZw2JJgZwL/uKL7q+q1Jc2YDmbMdLSekkHnh42HxfLSfPPsBjmGtyAniBoXe0Y/oWa584yWgR1na+Vo3hHH8tK0HJkgr6ccIQMlrmCHbUHGT+YRcP2ytn/VcV8Wzt7lWXN4x4qmE+PpK6+2iC8cHTwe6eQIDAQAB
-----END PUBLIC KEY-----`;

// 获取命令行传递的参数
const args = process.argv.slice(2);  // 第一个是 node，第二个是 JS 文件名，后面的就是传递的参数
const param1 = args[0];  // 获取第一个参数

// 用 PEM 格式的公钥加密数据
const data = param1;
const encryptedData = crypto.publicEncrypt(publicKeyPem, Buffer.from(data));

console.log(encryptedData.toString('base64'));
