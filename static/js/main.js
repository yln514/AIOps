// 通用JavaScript功能
document.addEventListener('DOMContentLoaded', function() {
    // 页面加载完成后执行
    console.log('页面加载完成');

    // 添加一些通用交互功能
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // 添加加载状态
    const loadingElements = document.querySelectorAll('.loading');
    loadingElements.forEach(el => {
        // 模拟加载完成后隐藏加载动画
        setTimeout(() => {
            el.style.display = 'none';
        }, 1500);
    });
});

// 工具函数
const utils = {
    // 格式化数字
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },

    // 格式化百分比
    formatPercentage: function(value) {
        return (value * 100).toFixed(2) + '%';
    },

    // 获取颜色渐变
    getColorGradient: function(baseColor, count) {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const alpha = 0.3 + (i * 0.7 / count);
            colors.push(`${baseColor}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`);
        }
        return colors;
    }
};

// API请求封装
const api = {
    // GET请求
    get: function(url, callback) {
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => callback(null, data))
            .catch(error => callback(error, null));
    },

    // POST请求
    post: function(url, data, callback) {
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => callback(null, data))
        .catch(error => callback(error, null));
    }
};
