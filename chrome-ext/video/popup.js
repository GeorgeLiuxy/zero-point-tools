// 定义全局变量存储时效信息

// 验证时效信息
function checkValidity() {

    const activationDate = new Date("2024-11-21");
    const currentDate = new Date();
    const timeElapsed = Math.floor((currentDate - activationDate) / (1000 * 60 * 60 * 24)); // 转为天数

    if (timeElapsed <= 30) {
        return true; // 授权有效
    } else {
        alert("授权已过期，请重新激活！");
        return false;
    }
}

// 播放/暂停按钮
document.getElementById('play-btn').addEventListener('click', function() {
    if (!checkValidity()) return; // 验证时效
    chrome.runtime.sendMessage({ action: 'playPause' }, function(response) {
        console.log('操作结果：', response.status);
    });
});

// 快进按钮（5秒）
document.getElementById('fast-forward-btn').addEventListener('click', function() {
    if (!checkValidity()) return; // 验证时效
    chrome.runtime.sendMessage({ action: 'fastForward' }, function(response) {
        console.log('操作结果：', response.status);
    });
});

// 倒退按钮（5秒）
document.getElementById('rewind-btn').addEventListener('click', function() {
    if (!checkValidity()) return; // 验证时效
    chrome.runtime.sendMessage({ action: 'rewind' }, function(response) {
        console.log('操作结果：', response.status);
    });
});

// 加速按钮（提高播放速率）
document.getElementById('speed-up-btn').addEventListener('click', function() {
    if (!checkValidity()) return; // 验证时效
    chrome.runtime.sendMessage({ action: 'speedUp' }, function(response) {
        console.log('操作结果：', response.status);
    });
});

// 减速按钮（降低播放速率）
document.getElementById('slow-down-btn').addEventListener('click', function() {
    if (!checkValidity()) return; // 验证时效
    chrome.runtime.sendMessage({ action: 'slowDown' }, function(response) {
        console.log('操作结果：', response.status);
    });
});

// 跳转到指定时间按钮
document.getElementById('jump-to-time-btn').addEventListener('click', function() {
    if (!checkValidity()) return; // 验证时效
    chrome.runtime.sendMessage({ action: 'jumpToTime' }, function(response) {
        console.log('操作结果：', response.status);
    });
});
