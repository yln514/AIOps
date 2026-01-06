// 图表配置对象
const chartConfig = {
    commonOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                font: {
                    size: 16
                }
            }
        }
    },

    pieOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
            },
            title: {
                display: true,
                font: {
                    size: 16
                }
            }
        }
    },

    barOptions: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            title: {
                display: true,
                font: {
                    size: 16
                }
            }
        },
        scales: {
            x: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: '数值'
                }
            }
        }
    }
};

// 渲染话单状态分布图表
function renderBillStatusChart(labels, data) {
    const ctx = document.getElementById('billStatusChart').getContext('2d');

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#1890FF',
                    '#52C41A',
                    '#FADB14',
                    '#FF7875',
                    '#722ED1',
                    '#FF9C6E',
                    '#B37FEB',
                    '#5CDBD3'
                ],
                borderWidth: 1
            }]
        },
        options: chartConfig.pieOptions
    });
}

// 渲染特征重要性图表
function renderFeatureImportanceChart(labels, data) {
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '特征重要性',
                data: data,
                backgroundColor: '#1890FF',
                borderColor: '#40A9FF',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            ...chartConfig.commonOptions,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: '特征重要性'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '重要性值'
                    }
                }
            }
        }
    });
}

// 渲染每小时异常分布图表
function renderHourlyAnomaliesChart(labels, data) {
    const ctx = document.getElementById('hourlyAnomaliesChart').getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '异常数量',
                data: data,
                borderColor: '#F5222D',
                backgroundColor: 'rgba(245, 34, 45, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            ...chartConfig.commonOptions,
            plugins: {
                title: {
                    display: true,
                    text: '每小时异常分布'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '异常数量'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '小时'
                    }
                }
            }
        }
    });
}

// 渲染根因分布图表
function renderRootCauseChart(labels, data) {
    const ctx = document.getElementById('rootCauseChart').getContext('2d');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#73C0DE',
                    '#3BA272',
                    '#FC8452',
                    '#9A60B4',
                    '#EA7CCC',
                    '#5470C6',
                    '#EE6666'
                ],
                borderWidth: 1
            }]
        },
        options: {
            ...chartConfig.pieOptions,
            cutout: '50%' // 创建环形图效果
        }
    });
}

// 渲染模型对比图表
function renderModelComparisonChart(labels, rfData, xgbData) {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Random Forest',
                    data: rfData,
                    backgroundColor: 'rgba(82, 196, 26, 0.7)',
                    borderColor: 'rgba(82, 196, 26, 1)',
                    borderWidth: 1
                },
                {
                    label: 'XGBoost',
                    data: xgbData,
                    backgroundColor: 'rgba(24, 144, 255, 0.7)',
                    borderColor: 'rgba(24, 144, 255, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...chartConfig.commonOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '分数'
                    }
                }
            }
        }
    });
}

// 更新指标卡片
function updateMetricsCards(metrics) {
    document.getElementById('accuracy-value').textContent = metrics.accuracy + '%';
    document.getElementById('precision-value').textContent = metrics.precision + '%';
    document.getElementById('recall-value').textContent = metrics.recall + '%';
    document.getElementById('f1-value').textContent = metrics.f1_score + '%';
    document.getElementById('total-records-value').textContent = utils.formatNumber(metrics.total_records);
    document.getElementById('anomaly-count-value').textContent = utils.formatNumber(metrics.anomaly_count);
    document.getElementById('normal-count-value').textContent = utils.formatNumber(metrics.normal_count);
}

// 初始化图表
function initializeCharts() {
    // 检查是否有图表容器
    if (document.getElementById('billStatusChart')) {
        // 获取图表数据
        api.get('/api/chart_data', function(error, data) {
            if (error) {
                console.error('获取图表数据失败:', error);
                return;
            }

            // 渲染各种图表
            renderBillStatusChart(data.bill_status.labels, data.bill_status.data);
            renderFeatureImportanceChart(data.feature_importance.labels, data.feature_importance.data);
            renderHourlyAnomaliesChart(data.hourly_anomalies.labels, data.hourly_anomalies.data);

            // 如果存在根因分布图表容器
            if (data.root_cause && data.root_cause.labels) {
                renderRootCauseChart(data.root_cause.labels, data.root_cause.data);
            }

            // 如果存在模型对比图表容器
            if (data.model_comparison) {
                renderModelComparisonChart(
                    data.model_comparison.labels,
                    data.model_comparison.rf_data,
                    data.model_comparison.xgb_data
                );
            }
        });
    }

    // 获取并更新指标
    if (document.querySelector('.metric-value')) {
        api.get('/api/metrics', function(error, metrics) {
            if (error) {
                console.error('获取指标数据失败:', error);
                return;
            }

            updateMetricsCards(metrics);
        });
    }
}

// 页面加载完成后初始化图表
document.addEventListener('DOMContentLoaded', initializeCharts);

// 页面显示时重新调整图表大小
window.addEventListener('pageshow', function() {
    // 延迟执行以确保DOM完全加载
    setTimeout(initializeCharts, 100);
});
