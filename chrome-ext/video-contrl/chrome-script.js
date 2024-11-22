(function() {
    // 检查页面是否有视频元素
    const video = document.querySelector('video');
    if (!video) {
        console.log('未找到视频元素');
        return;
    }

    // 创建控制面板
    const controlsPanel = document.createElement('div');
    controlsPanel.innerHTML = `
        <button id="play-btn">播放/暂停</button>
        <button id="fast-forward-btn">快进</button>
        <button id="rewind-btn">倒退</button>
        <button id="speed-up-btn">加速</button>
        <button id="slow-down-btn">减速</button>
        <button id="jump-to-time-btn">跳转到时间</button>
    `;

    // 设置控制面板样式
    controlsPanel.style.position = 'fixed';
    controlsPanel.style.top = '20px';
    controlsPanel.style.right = '20px';
    controlsPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';
    controlsPanel.style.color = 'white';
    controlsPanel.style.padding = '10px';
    controlsPanel.style.borderRadius = '8px';
    controlsPanel.style.display = 'flex';
    controlsPanel.style.flexDirection = 'column';
    controlsPanel.style.gap = '10px';
    controlsPanel.style.zIndex = 1000;

    // 将控制面板添加到页面中
    document.body.appendChild(controlsPanel);

    // 播放/暂停按钮
    document.getElementById('play-btn').addEventListener('click', function() {
        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
    });

    // 快进按钮（5秒）
    document.getElementById('fast-forward-btn').addEventListener('click', function() {
        video.currentTime += 5;
    });

    // 倒退按钮（5秒）
    document.getElementById('rewind-btn').addEventListener('click', function() {
        video.currentTime -= 5;
    });

    // 加速按钮（提高播放速率）
    document.getElementById('speed-up-btn').addEventListener('click', function() {
        video.playbackRate += 0.25;
    });

    // 减速按钮（降低播放速率）
    document.getElementById('slow-down-btn').addEventListener('click', function() {
        video.playbackRate -= 0.25;
    });

    // 跳转到指定时间按钮
    document.getElementById('jump-to-time-btn').addEventListener('click', function() {
        const jumpTime = prompt("请输入要跳转的时间（秒）：");
        if (jumpTime !== null && !isNaN(jumpTime)) {
            video.currentTime = parseFloat(jumpTime);
        }
    });
})();