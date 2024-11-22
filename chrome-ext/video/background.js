// 监听来自 popup.js 或其他地方的消息
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'playPause') {
        // 向当前活动标签页发送指令，控制视频播放/暂停
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            chrome.tabs.executeScript(tabs[0].id, {
                code: `
                    var video = document.querySelector('video');
                    if (video) {
                        if (video.paused) {
                            video.play();
                            console.log('视频播放');
                        } else {
                            video.pause();
                            console.log('视频暂停');
                        }
                    }
                `
            });
        });
    }
    if(request.action === 'fastForward'){
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            chrome.tabs.executeScript(tabs[0].id, {
                code: `
                    var video = document.querySelector('video');
                    if (video) {
                       video.currentTime += 5;
                    }
                `
            });
        });
    }
    if(request.action === 'rewind'){
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            chrome.tabs.executeScript(tabs[0].id, {
                code: `
                    var video = document.querySelector('video');
                    if (video) {
                       video.currentTime -= 5;
                    }
                `
            });
        });
    }
    if(request.action === 'speedUp'){
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            chrome.tabs.executeScript(tabs[0].id, {
                code: `
                    var video = document.querySelector('video');
                    if (video) {
                       video.playbackRate += 0.25;
                    }
                `
            });
        });
    }
    if(request.action === 'slowDown'){
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            chrome.tabs.executeScript(tabs[0].id, {
                code: `
                    var video = document.querySelector('video');
                    if (video) {
                       video.playbackRate -= 0.25;
                    }
                `
            });
        });
    }
    if(request.action === 'jumpToTime'){
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            chrome.tabs.executeScript(tabs[0].id, {
                code: `
                    var video = document.querySelector('video');
                    if (video) {
                       const jumpTime = prompt("请输入要跳转的时间（秒）：");
                        if (jumpTime !== null && !isNaN(jumpTime)) {
                            video.currentTime = parseFloat(jumpTime);
                        }
                    }
                `
            });
        });
    }

    sendResponse({ status: 'success' });

});
