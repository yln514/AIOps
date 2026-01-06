import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix,
    classification_report, precision_score, make_scorer
)
import xgboost as xgb


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ==============================
# 模块 1：数据加载与探索性分析（EDA）
# ==============================
def load_and_explore_data(file_path="telecom_bill_data.csv"):
    """加载数据并进行基础探索"""
    df = pd.read_csv(file_path, encoding='utf-8')
    print("=== 数据基本信息 ===")
    print(f"总行数: {df.shape[0]}")
    print(f"字段类型:\n{df.dtypes}")
    print(f"缺失值占比:\n{df.isnull().mean() * 100}")

    # 1.b bill_status 分布比例柱状图
    plt.figure()
    type_bill_status = df['bill_status'].value_counts().index.tolist()  # 话单状态索引列表
    count_bill_status = np.array(df['bill_status'].value_counts().values.tolist())  # 对应的数量列表
    rate_bill_status = np.round((count_bill_status/10000)*100,2)  # 计算比例, 保留两位小数
    plt.bar(type_bill_status, rate_bill_status)
    plt.title('话单状态分布比例')
    plt.xlabel('话单状态')
    plt.ylabel('比例 (%)')
    plt.ylim(0, 100)
    # 在每个数据点上显示具体数值
    for x, y in zip(type_bill_status, rate_bill_status):
        print(x, y)
        plt.text(x, y, str(y), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('output/bill_status分布比例柱状图.png')
    plt.show()

    # 1.c drop_rate 和 signal_strength 箱线图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制第一张图：drop_rate 箱型图
    axes[0].boxplot(df['drop_rate'])
    axes[0].set_title('基站掉话率分布')
    axes[0].set_ylabel('掉话率')
    axes[0].grid(True, alpha=0.5, axis='y', linestyle='--')

    # 绘制第二张图：signal_strength 箱型图
    axes[1].boxplot(df['signal_strength'])
    axes[1].set_title('信号强度分布')
    axes[1].set_ylabel('信号强度')
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

    # 确保两个子图大小一致
    axes[0].set_ylim(df['drop_rate'].min() - abs(df['drop_rate'].min()) * 0.1,
                     df['drop_rate'].max() + df['drop_rate'].max() * 0.1)
    axes[1].set_ylim(df['signal_strength'].min() - abs(df['signal_strength'].min()) * 0.1,
                     df['signal_strength'].max() + df['signal_strength'].max() * 0.1)

    plt.tight_layout()
    plt.savefig('output/基站掉话率和信号强度箱型图.png')
    plt.show()

    # 打印统计特征
    print("\n=== 数值特征统计 ===")
    print(df[['drop_rate', 'signal_strength']].describe())

    return df

# ==============================
# 模块 2：数据预处理
# ==============================
def preprocess_data(df):
    """完成清洗、编码、标准化、特征选择"""
    df = df.copy()

    # 2.1 处理重复值（按 user_id + create_hour + call_duration）
    df.drop_duplicates(subset=['user_id', 'create_hour', 'call_duration'], inplace=True)
    print(f"去重后数据量: {df.shape[0]}")

    # 2.2 处理明显错误的异常值（非业务异常）
    df = df[df['call_duration'] >= 0]  # 删除负通话时长
    # 注意：cost > 10 是业务异常（费用异常），不应删除！

    # 2.3 缺失值处理
    if df['cost'].isnull().any():
        # 费用缺失值用均值填充
        df['cost'].fillna(df['cost'].mean(), inplace=True)
    if df['base_station'].isnull().any():
        # 基站缺失值用众数填充
        df['base_station'].fillna(df['base_station'].mode()[0], inplace=True)
    # 对其他字段出现缺失值则删除整行
    df.dropna(inplace=True)

    # 2.4 特征编码
    le_base = LabelEncoder()
    df['base_station_encoded'] = le_base.fit_transform(df['base_station'])

    le_status = LabelEncoder()
    df['bill_status_encoded'] = le_status.fit_transform(df['bill_status'])

    # 2.5 特征标准化
    scaler = StandardScaler()
    numeric_cols = ['call_duration', 'cost', 'signal_strength', 'drop_rate']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 2.6 特征选择（使用随机森林评估重要性）
    X_feat = df[['call_duration', 'cost', 'signal_strength', 'drop_rate',
                 'base_station_encoded', 'create_hour']]
    y_feat = (df['bill_status'] != '正常').astype(int)  # 异常标签

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_feat, y_feat)
    importances = rf_selector.feature_importances_
    feature_names = X_feat.columns

    # 绘制特征重要性（提前用于可视化模块）
    plt.figure(figsize=(8, 5))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('特征重要性')
    plt.ylim(0, 0.3)
    for x, y in zip(range(len(importances)), importances[indices]):
        print(x, y)
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')  # 保留两位小数
    plt.tight_layout()
    plt.savefig('output/特征重要性柱状图.png')
    plt.show()

    # 返回处理后的 DataFrame 和特征矩阵
    # 仅选择重要特征进行异常检测
    selected_features = ['call_duration', 'cost', 'signal_strength', 'drop_rate']
    X = df[selected_features]
    y_true = (df['bill_status'] != '正常').astype(int)

    return df, X, y_true


# ==============================
# 模块 3（GridSearchCV 版）：使用 GridSearchCV 调优 Isolation Forest
# ==============================
class IsoForestWrapper(BaseEstimator):
    """
    包装 IsolationForest，使其接口符合 scikit-learn 分类器规范：
    - fit(X, y=None)：y 被忽略（无监督）
    - predict(X)：返回 0（正常）或 1（异常）
    """
    def __init__(self, n_estimators=100, contamination=0.1, max_samples='auto', random_state=None):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.iso_ = None

    def fit(self, X, y=None):
        """训练孤立森林（忽略 y）"""
        self.iso_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iso_.fit(X)
        return self

    def predict(self, X):
        """预测：-1 → 1（异常），1 → 0（正常）"""
        preds = self.iso_.predict(X)
        return (preds == -1).astype(int)

def detect_anomalies_with_gridsearch(X, y_true):
    """
    使用 GridSearchCV 对 IsolationForest 进行超参数调优
    参考 scikit-learn 课件中 GridSearchCV 的标准用法
    """
    # 1. 定义超参数网格
    param_grid = {
        # 影响不大
        'n_estimators': [50, 100, 200],
        # contamination范围:0-0.5, 值设置小为了高准确度(少误判),值设置大为了少漏报
        'contamination': [0.1, 0.15, 0.2],
        'max_samples': ['auto', 256]
    }

    # 2. 创建模型实例（带随机种子）
    iso_wrapper = IsoForestWrapper(random_state=42)

    # 3. 定义评分器（以 F1 为目标，因异常检测需平衡精确率与召回率）
    scoring = make_scorer(f1_score)

    # 4. 配置 GridSearchCV
    grid_search = GridSearchCV(
        estimator=iso_wrapper,
        param_grid=param_grid,
        scoring=scoring,
        cv=3,                # 3折交叉验证（基于 y_true 评估）
        n_jobs=-1,
        verbose=1
    )

    # 5. 执行搜索（注意：fit 中 y_true 仅用于 CV 评分，不参与模型训练）
    grid_search.fit(X, y_true)

    # 6. 获取最佳模型和预测结果
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)

    # # 7. 输出结果
    # print(f"\n 最佳超参数: {grid_search.best_params_}")
    # print(f" 最佳交叉验证 F1 分数: {grid_search.best_score_:.4f}")

    # 8. 完整评估指标
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== 调优后异常检测效果（在全量数据上） ===")

    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"精确率 (Precision): {pre:.4f}")
    print(f"召回率 (Recall):   {rec:.4f}")
    print(f"F1 分数:           {f1:.4f}")

    # 9. 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '异常'],
                yticklabels=['正常', '异常'])
    plt.title('异常检测混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('output/异常检测混淆矩阵.png')
    plt.show()

    return y_pred, best_model, grid_search.best_params_

# ==============================
# 模块 4：可视化分析
# ==============================
def visualize_analysis(df, y_pred):
    """完成三项可视化任务"""
    df_vis = df.copy()
    df_vis['is_anomaly'] = y_pred

    # 4.1 根因分布饼图（双子图展示）
    anomaly_causes = df_vis[df_vis['is_anomaly'] == 1]['root_cause']
    cause_counts_all = anomaly_causes.value_counts()

    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1：显示"无异常"与"有异常"的总体分布（突出显示"有异常"）

    # 分离"无异常"和其他异常
    no_anomaly_count = cause_counts_all['无异常']
    other_anomalies_count = cause_counts_all.sum() - no_anomaly_count

    # 绘制总体分布饼图，突出显示"有异常"
    overall_counts = [no_anomaly_count, other_anomalies_count]
    overall_labels = ['无异常', '有异常']
    explode = (0, 0.1)  # 突出显示"有异常"部分
    ax1.pie(overall_counts,
            labels=overall_labels,
            autopct='%1.1f%%',
            startangle=140,
            explode=explode,
            colors = ['#0868ac', '#43a2ca'])
    ax1.set_title('异常总体分布')

    # 子图2：显示排除"无异常"后的各类具体异常占比
    filtered_causes = anomaly_causes[anomaly_causes != '无异常']
    cause_counts_filtered = filtered_causes.value_counts()

    ax2.pie(cause_counts_filtered,
            labels=cause_counts_filtered.index,
            autopct='%1.1f%%',
            startangle=140,
            colors = ['#8971d0','#7dace4','#95e8d7','#adf7d1','#fff4e1']
            )
    ax2.set_title('具体异常类型分布（排除无异常）')


    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('output/根因分布饼图.png')
    plt.show()

    # 4.2 按小时统计异常数量折线图
    hourly_anomalies = df_vis[df_vis['is_anomaly'] == 1].groupby('create_hour').size()
    all_hours = pd.Series(0, index=range(24))
    hourly_anomalies = all_hours.add(hourly_anomalies, fill_value=0)

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_anomalies.index, hourly_anomalies.values, marker='o', color='green')
    plt.title('每小时异常话单数量变化')
    plt.xlabel('小时')
    plt.ylabel('异常数量')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/异常话单小时分布折线图.png')
    plt.show()

# ==============================
# 模块 5：极端异常样本分析
# ==============================
def analyze_extreme_anomalies(df_original, iso_model, X):
    """
    分析极端异常样本的分布特征
    参数:
        df_original: 原始处理后的 DataFrame（包含 root_cause, base_station 等字段）
        iso_model: 训练好的 IsolationForest 模型（或 IsoForestWrapper）
        X: 特征矩阵（与 df_original 行对齐）
    """
    # 获取预测标签和异常分数
    if hasattr(iso_model, 'iso_'):  # 如果是 Wrapper
        y_pred = iso_model.predict(X)
        anomaly_scores = iso_model.iso_.decision_function(X)
    else:  # 如果是原始 IsolationForest
        y_pred = iso_model.predict(X)
        anomaly_scores = iso_model.decision_function(X)

    # 只保留异常样本（y_pred == -1 或 1？注意：Wrapper 输出 0/1，原始模型输出 -1/1）
    # 为了通用性，我们统一用原始模型的 -1/1 判断
    if hasattr(iso_model, 'iso_'):
        # Wrapper 的底层模型
        y_pred_raw = iso_model.iso_.predict(X)
    else:
        y_pred_raw = y_pred

    # 提取异常样本
    anomaly_mask = (y_pred_raw == -1)
    df_anomalies = df_original[anomaly_mask].copy()
    scores_anomalies = anomaly_scores[anomaly_mask]

    if len(df_anomalies) == 0:
        print("未检测到任何异常样本，无法分析极端异常。")
        return

    # 定义“极端异常”：异常样本中 anomaly_score 最低的 10%（即最异常的）
    threshold = np.percentile(scores_anomalies, 10)  # 第10百分位（更小=更异常）
    extreme_mask = scores_anomalies <= threshold
    df_extreme = df_anomalies[extreme_mask]

    print("===极端异常样本分析（最异常的 10% 异常样本）===")
    print(f"总异常样本数: {len(df_anomalies)}")
    print(f"极端异常样本数: {len(df_extreme)} ({len(df_extreme)/len(df_anomalies):.1%} of anomalies)")

    # 1. 根因分布
    print("\n根因（root_cause）分布:")
    cause_dist = df_extreme['root_cause'].value_counts()
    for cause, count in cause_dist.items():
        print(f"  {cause}: {count} 条 ({count/len(df_extreme):.1%})")

    # 2. 基站分布
    print("\n归属基站（base_station）分布:")
    bs_dist = df_extreme['base_station'].value_counts()
    for bs, count in bs_dist.items():
        print(f"  {bs}: {count} 条 ({count/len(df_extreme):.1%})")

    # 3. 异常高峰小时
    print("\n异常话单生成小时分布（Top 5 高峰时段）:")
    hour_dist = df_extreme['create_hour'].value_counts().sort_values(ascending=False).head(5)
    for hour, count in hour_dist.items():
        print(f"  {hour:02d}:00 - {hour+1:02d}:00: {count} 条")

    # 4. 数值特征统计（可选）
    print("\n极端异常样本数值特征均值:")
    num_cols = ['call_duration', 'cost', 'signal_strength', 'drop_rate']
    for col in num_cols:
        if col in df_extreme.columns:
            print(f"  {col}: {df_extreme[col].mean():.2f}")

# ==============================
# 拓展任务：根因分类模型对比（Random Forest vs XGBoost）
# ==============================
def compare_root_cause_classifiers(df_processed):
    """
    在异常话单上训练 Random Forest 和 XGBoost，预测 root_cause
    """
    # 1. 准备数据：只用异常样本
    df_anomalies = df_processed[df_processed['bill_status'] != '正常'].copy()
    if len(df_anomalies) == 0:
        print("无异常样本，无法进行根因分类。")
        return

    # 2. 特征与标签
    feature_cols = ['call_duration', 'cost', 'signal_strength', 'drop_rate',
                    'base_station_encoded', 'create_hour']
    X_rc = df_anomalies[feature_cols]
    y_rc = df_anomalies['root_cause']

    # 3. 对标签进行编码
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_rc_encoded = label_encoder.fit_transform(y_rc)

    # 4. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_rc, y_rc_encoded, test_size=0.3, random_state=42, stratify=y_rc_encoded
    )

    # print(f"\n根因分类任务：共 {len(df_anomalies)} 条异常样本，{len(label_encoder.classes_)} 类根因")

    # ==============================
    # 模型1：Random Forest
    # ==============================
    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    y_pred_rf = rf_grid.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')

    # ==============================
    # 模型2：XGBoost
    # ==============================
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1]
    }
    xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0)
    xgb_grid.fit(X_train, y_train)
    y_pred_xgb = xgb_grid.predict(X_test)

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')

    # ==============================
    # 结果对比与详细报告
    # ==============================
    print("---两种分类模型对比结果---")
    print(f"{'模型':<15} {'准确率':<10} {'F1-macro':<10}")
    print(f"{'-'*40}")
    print(f"{'Random Forest':<15} {acc_rf:<10.4f} {f1_rf:<10.4f}")
    print(f"{'XGBoost':<15} {acc_xgb:<10.4f} {f1_xgb:<10.4f}")


# ==============================
# 模块 6：主函数（执行流程）
# ==============================
def main():
    # 创建输出目录（可手动创建，或用 os.makedirs）
    import os
    os.makedirs('output', exist_ok=True)

    # 步骤1：加载与探索
    df = load_and_explore_data()

    # 步骤2：预处理
    df_processed, X, y_true = preprocess_data(df)

    # 步骤3：使用 GridSearchCV 调优的异常检测
    y_pred, best_model, best_params = detect_anomalies_with_gridsearch(X, y_true)

    # 步骤4：可视化
    visualize_analysis(df_processed, y_pred)

    # 步骤5：分析极端异常样本
    analyze_extreme_anomalies(df_processed, best_model, X)

    # 步骤6：两种模型(Random Forest VS XGBoost)对比
    compare_root_cause_classifiers(df_processed)

    print("\n 所有任务已完成！图表已保存至 output/ 目录。")


if __name__ == "__main__":
    main()