const fs = require('fs');
const path = require('path');
const JavaScriptObfuscator = require('javascript-obfuscator');

// 指定需要混淆的文件夹
const inputFolder = './video'; // 你的 JS 文件所在目录
const outputFolder = './video_ctl_1'; // 混淆后的文件保存到同一个目录

// 遍历文件夹中的所有文件
fs.readdir(inputFolder, (err, files) => {
    if (err) {
        console.error('读取文件夹失败：', err);
        return;
    }

    files.forEach(file => {
        const filePath = path.join(inputFolder, file);

        // 仅处理 .js 文件
        if (path.extname(file) === '.js') {
            console.log(`正在混淆文件：${filePath}`);

            // 读取文件内容
            const jsContent = fs.readFileSync(filePath, 'utf-8');

            // 混淆文件内容
            const obfuscatedContent = JavaScriptObfuscator.obfuscate(jsContent, {
                compact: true,
                controlFlowFlattening: true, // 控制流平坦化，增加代码复杂性
                deadCodeInjection: true,    // 注入无用代码
                debugProtection: true,     // 防调试
                stringArrayEncoding: ['base64'], // 字符串数组编码
                disableConsoleOutput: true // 禁用 console 输出
            }).getObfuscatedCode();

            // 写入混淆后的文件
            const outputFilePath = path.join(outputFolder, file);
            fs.writeFileSync(outputFilePath, obfuscatedContent, 'utf-8');
            console.log(`文件已混淆：${outputFilePath}`);
        }
    });
});
