from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import os
from B23011914_严磊_AIOps作业 import load_and_explore_data, preprocess_data, detect_anomalies_with_gridsearch, \
    visualize_analysis, analyze_extreme_anomalies, compare_root_cause_classifiers

app = Flask(__name__)

# 加载数据的全局变量
df = None
df_processed = None
X = None
y_true = None
y_pred = None
best_model = None


@app.route('/')
def index():
    """主页路由 - 显示首页"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """仪表盘页面"""
    # 如果数据尚未加载，则加载数据
    load_data_if_needed()

    # 准备要传递给前端的数据
    metrics = calculate_metrics()
    return render_template('dashboard.html', metrics=metrics)


@app.route('/analysis')
def analysis():
    """分析详情页面"""
    load_data_if_needed()
    return render_template('analysis.html')


@app.route('/api/metrics')
def get_metrics():
    """获取关键指标的API端点"""
    load_data_if_needed()
    metrics = calculate_metrics()
    return jsonify(metrics)


@app.route('/api/chart_data')
def get_chart_data():
    """获取图表数据的API端点"""
    load_data_if_needed()

    # 验证必要的数据是否存在
    if df is None:
        return jsonify({'error': '数据未加载'}), 500

    if df.empty:
        return jsonify({'error': '数据为空'}), 500

    # 验证所需的列是否存在
    required_columns = ['bill_status', 'call_duration', 'cost', 'signal_strength',
                        'drop_rate', 'base_station_encoded', 'create_hour']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return jsonify({'error': f'缺少必要的数据列: {missing_columns}'}), 500

    try:
        # 话单状态分布数据
        bill_status_counts = df['bill_status'].value_counts().to_dict()

        # 特征重要性数据
        from sklearn.ensemble import RandomForestClassifier
        X_feat = df[['call_duration', 'cost', 'signal_strength', 'drop_rate',
                     'base_station_encoded', 'create_hour']]
        y_feat = (df['bill_status'] != '正常').astype(int)

        # 检查是否有足够的数据进行训练
        if len(X_feat) == 0 or len(y_feat) == 0:
            return jsonify({'error': '没有足够的数据进行特征重要性分析'}), 500

        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X_feat, y_feat)
        importances = rf_selector.feature_importances_
        feature_names = X_feat.columns.tolist()

        # 检查 df_processed 是否存在
        if df_processed is None:
            return jsonify({'error': '处理后的数据未加载'}), 500

        # 按小时异常分布数据
        hourly_anomalies = df_processed[df_processed['is_anomaly'] == 1].groupby('create_hour').size()
        all_hours = pd.Series(0, index=range(24))
        hourly_anomalies = all_hours.add(hourly_anomalies, fill_value=0)

        chart_data = {
            'bill_status': {
                'labels': list(bill_status_counts.keys()),
                'data': list(bill_status_counts.values())
            },
            'feature_importance': {
                'labels': feature_names,
                'data': importances.tolist()
            },
            'hourly_anomalies': {
                'labels': list(range(24)),
                'data': hourly_anomalies.values.tolist()
            }
        }

        return jsonify(chart_data)
    except Exception as e:
        print(f"Error in get_chart_data: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis_data')
def get_analysis_data():
    """获取分析详情页面的数据"""
    load_data_if_needed()

    # 构建分析数据 - 这里使用示例数据，实际应从处理后的数据中获取
    analysis_data = {
        'anomaly_distribution': {
            'labels': ['正常', '费用异常', '信号异常', '基站异常', '时长异常'],
            'data': [7500, 800, 600, 500, 400]  # 示例数据
        },
        'root_cause': {
            'labels': ['设备故障', '网络拥塞', '信号干扰', '基站异常', '其他'],
            'data': [40, 25, 15, 12, 8]  # 百分比数据
        },
        'extreme_anomaly': {
            'labels': ['基站A', '基站B', '基站C', '基站D', '基站E'],
            'data': [35, 25, 20, 12, 8]
        },
        'model_comparison': {
            'labels': ['准确率', '精确率', '召回率', 'F1分数'],
            'rf_data': [92.5, 89.2, 91.8, 90.5],
            'xgb_data': [93.1, 90.1, 92.3, 91.2]
        }
    }

    return jsonify(analysis_data)


def load_data_if_needed():
    """如果数据未加载，则加载数据"""
    global df, df_processed, X, y_true, y_pred, best_model

    if df is None:
        try:
            df = load_and_explore_data()
            if df is None or df.empty:
                raise Exception("数据加载失败或数据为空")

            df_processed, X, y_true = preprocess_data(df)
            y_pred, best_model, best_params = detect_anomalies_with_gridsearch(X, y_true)

            # 添加预测结果到数据框
            df_processed['is_anomaly'] = y_pred
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈
            # 创建示例数据以避免应用崩溃
            df = pd.DataFrame({
                'bill_status': ['正常', '异常', '正常', '异常'],
                'call_duration': [100, 200, 150, 300],
                'cost': [10, 20, 15, 30],
                'signal_strength': [-70, -80, -75, -85],
                'drop_rate': [0.1, 0.2, 0.15, 0.25],
                'base_station_encoded': [1, 2, 1, 3],
                'create_hour': [10, 11, 12, 13],
                'is_anomaly': [0, 1, 0, 1]
            })
            df_processed = df.copy()


def calculate_metrics():
    """计算关键指标"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if y_pred is not None and y_true is not None:
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            # 计算异常数量
            total_records = len(y_true)
            anomaly_count = sum(y_pred)
            normal_count = total_records - anomaly_count

            return {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'total_records': total_records,
                'anomaly_count': anomaly_count,
                'normal_count': normal_count
            }
        except Exception as e:
            print(f"Error in calculate_metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    return {}


if __name__ == '__main__':
    app.run(debug=True)
