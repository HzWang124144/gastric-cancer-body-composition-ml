from scipy.stats import pearsonr

# 分离特征和目标变量
X = df_imputed.drop(columns=['STATUS3Y'])
y = df_imputed['STATUS3Y']
#%%
# 计算每个特征与目标变量的皮尔逊相关系数
correlations = {}
for col in X.columns:
    corr, _ = pearsonr(X[col], y)
    correlations[col] = corr

# 将结果转换为DataFrame并按相关系数绝对值排序
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['PearsonR'])
corr_df['absR'] = corr_df['PearsonR'].abs()
corr_df.sort_values(by='absR', ascending=False, inplace=True)

# 显示相关性排名前10的特征
print(corr_df.head(10))

# ============================
# 计算VIF（方差膨胀因子）
# ============================
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 添加常数项用于VIF计算
X_with_const = add_constant(X)

# 计算每个特征的VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                   for i in range(len(X_with_const.columns))]

# 移除常数项的VIF（通常为无穷大或很大，没有实际意义）
vif_data = vif_data[vif_data['feature'] != 'const']

# 按VIF值降序排序
vif_data = vif_data.sort_values(by='VIF', ascending=False)

# 显示VIF排名前10的特征
print("VIF最高的前10个特征：")
print(vif_data.head(10))

# 显示VIF>10的特征（通常认为VIF>10存在严重多重共线性）
high_vif_features = vif_data[vif_data['VIF'] > 10]
if not high_vif_features.empty:
    print(f"\n存在多重共线性的特征（VIF > 10）：")
    print(high_vif_features)
else:
    print(f"\n没有特征的VIF大于10，多重共线性问题不大。")

# 显示VIF>5的特征（中度多重共线性）
moderate_vif_features = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
if not moderate_vif_features.empty:
    print(f"\n存在中度多重共线性的特征（5 < VIF ≤ 10）：")

    print(moderate_vif_features)

# 计算特征之间的相关系数矩阵
corr_matrix = df_imputed[['STATUS3Y']+list(X.columns)].corr()

# 绘制热力图
plt.figure(figsize=(32, 32))
sns.heatmap(corr_matrix,cmap='coolwarm', 
            center=0.5,
            annot=True, fmt='.2f', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()



import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings
import time
warnings.filterwarnings('ignore')

# 确保使用正确的训练数据
print(f"训练数据形状: {X_train_scaled.shape}")
print(f"特征名称: {feature_names[:5]}... (共{len(feature_names)}个)")

# 1. 定义嵌套交叉验证策略
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

# 2. 定义CatBoost参数网格（针对小数据集优化）
catboost_param_grid = {
    'iterations': [100, 200, 300, 500],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.001, 0.01, 0.1, 0.3],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128],
    'random_strength': [0, 0.1, 1],
    'bagging_temperature': [0, 0.5, 1],
    'od_type': ['IncToDec', 'Iter'],
    'od_wait': [10, 20, 50],
    'loss_function': ['Logloss'],
    'eval_metric': ['AUC'],
    'random_seed': [42]
}

# 3. 自定义CatBoost包装器，提供feature_importances_属性供RFECV使用
class CatBoostWrapper(CatBoostClassifier):
    """
    包装CatBoostClassifier，通过get_feature_importance()提供feature_importances_属性。
    """
    @property
    def feature_importances_(self):
        if hasattr(self, 'get_feature_importance'):
            return self.get_feature_importance()
        else:
            # 模型未拟合时，RFECV不会调用此属性
            raise AttributeError("Model not fitted yet")

    def fit(self, X, y):
        # 直接调用父类fit，无需手动设置n_features_in_
        return super().fit(X, y, verbose=False)

# 4. 初始化记录器
outer_scores = []
all_selected_features = []
best_inner_models = []
fold_times = []

print(f"开始进行 {outer_cv.n_splits} 折外层交叉验证...")

# 5. 外层循环
for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_scaled, y_train)):
    fold_start_time = time.time()
    print(f"\n--- 外层折 {fold+1}/{outer_cv.n_splits} ---")

    X_outer_train, X_outer_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_outer_train, y_outer_val = y_train[train_idx], y_train[val_idx]

    # 创建基础模型（用于RFECV）
    base_catboost = CatBoostWrapper(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False
    )

    # Step A: RFECV特征选择
    print(f"    进行递归特征消除(RFECV)...")
    rfecv = RFECV(
        estimator=clone(base_catboost),
        step=1,
        cv=inner_cv,
        scoring='roc_auc',
        min_features_to_select=5,
        n_jobs=1,
        verbose=0
    )

    rfecv.fit(X_outer_train, y_outer_train)

    selected_feature_mask = rfecv.support_
    selected_indices = np.where(selected_feature_mask)[0].tolist()
    all_selected_features.extend(selected_indices)

    X_outer_train_selected = X_outer_train[:, selected_feature_mask]
    X_outer_val_selected = X_outer_val[:, selected_feature_mask]

    n_selected = len(selected_indices)
    print(f"        RFECV选择特征数: {n_selected}")
    best_rfecv_score = np.max(rfecv.cv_results_['mean_test_score'])
    print(f"        RFECV最佳CV AUC: {best_rfecv_score:.4f}")

    # Step B: 超参数调优
    print(f"    在选定的 {n_selected} 个特征上进行超参数搜索...")
    inner_search = RandomizedSearchCV(
        estimator=CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=False
        ),
        param_distributions=catboost_param_grid,
        n_iter=30,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42 + fold,
        verbose=0
    )

    inner_search.fit(X_outer_train_selected, y_outer_train)
    best_inner_model = inner_search.best_estimator_
    best_inner_models.append(best_inner_model)

    # 评估外层验证集
    y_val_pred_proba = best_inner_model.predict_proba(X_outer_val_selected)[:, 1]
    fold_score = roc_auc_score(y_outer_val, y_val_pred_proba)
    outer_scores.append(fold_score)

    fold_time = time.time() - fold_start_time
    fold_times.append(fold_time)

    print(f"    外层验证集 AUC: {fold_score:.4f}")
    print(f"    最佳参数: {list(inner_search.best_params_.items())[:3]}...")
    print(f"    本折耗时: {fold_time:.1f} 秒")

# 6. 汇总结果
mean_outer_score = np.mean(outer_scores)
std_outer_score = np.std(outer_scores)
print(f"\n{'='*60}")
print("嵌套交叉验证结果总结")
print(f"{'='*60}")
print(f"平均外层AUC: {mean_outer_score:.4f} (±{std_outer_score:.4f})")
print(f"各折AUC详情: {[f'{s:.4f}' for s in outer_scores]}")
print(f"总耗时: {sum(fold_times):.1f} 秒 | 平均每折: {np.mean(fold_times):.1f} 秒")

# 7. 特征选择稳定性分析
print(f"\n{'='*60}")
print("特征选择稳定性分析")
print(f"{'='*60}")

from collections import Counter
feature_counter = Counter(all_selected_features)
n_outer_folds = outer_cv.n_splits

print(f"特征在 {n_outer_folds} 折外层CV中被选中的次数:")
print("-" * 50)

feature_stability = []
for feat_idx in range(len(feature_names)):
    count = feature_counter.get(feat_idx, 0)
    freq = count / n_outer_folds
    feature_stability.append((feat_idx, feature_names[feat_idx], count, freq))

feature_stability.sort(key=lambda x: x[2], reverse=True)

for feat_idx, feat_name, count, freq in feature_stability:
    if count > 0:
        print(f"  特征[{feat_idx:2d}] {feat_name:20s}: 选中 {count} 次, 频率 {freq:.1%}")

# 8. 确定最终特征子集
print(f"\n{'='*60}")
print("确定最终特征子集")
print(f"{'='*60}")

stability_threshold = 0.2
stable_core_features = [feat_idx for feat_idx, _, _, freq in feature_stability if freq >= stability_threshold]

print(f"稳定核心特征（在≥{stability_threshold:.0%}折中被选中）: {len(stable_core_features)} 个")
if stable_core_features:
    print("稳定特征索引:", stable_core_features)
    print("稳定特征名称:", [feature_names[i] for i in stable_core_features])
else:
    print("警告：没有特征达到稳定性阈值！将使用最常被选中的特征。")
    stable_core_features = [feat_idx for feat_idx, _, _, _ in feature_stability[:5]]

min_desired_features = 5
final_feature_indices = stable_core_features.copy()
if len(final_feature_indices) < min_desired_features:
    additional_needed = min_desired_features - len(final_feature_indices)
    remaining_features = [feat_idx for feat_idx, _, count, _ in feature_stability
                         if feat_idx not in final_feature_indices and count > 0]
    final_feature_indices.extend(remaining_features[:additional_needed])

print(f"\n最终特征子集 ({len(final_feature_indices)} 个):")
print("索引:", final_feature_indices)
print("名称:", [feature_names[i] for i in final_feature_indices])

# 9. 训练最终模型
print(f"\n{'='*60}")
print("在完整训练集上训练最终CatBoost模型")
print(f"{'='*60}")

X_train_final = X_train_scaled[:, final_feature_indices]
final_params = best_inner_models[-1].get_params()
final_params['verbose'] = False

print(f"使用 {X_train_final.shape[1]} 个特征训练最终模型...")
final_catboost = CatBoostClassifier(**final_params)
final_catboost.fit(X_train_final, y_train, verbose=False)

# 评估最终模型
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(final_catboost, X_train_final, y_train,
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                           scoring='roc_auc', n_jobs=1)
print(f"最终模型在训练集上的CV AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# 10. 保存结果
catboost_ncv_results = {
    'final_model': final_catboost,
    'final_feature_indices': final_feature_indices,
    'final_feature_names': [feature_names[i] for i in final_feature_indices],
    'outer_scores': outer_scores,
    'mean_outer_score': mean_outer_score,
    'std_outer_score': std_outer_score,
    'feature_stability': feature_stability,
    'all_selected_features_history': all_selected_features,
    'best_inner_models': best_inner_models,
    'feature_counter': feature_counter
}

print(f"\n{'='*80}")
print("CatBoost嵌套交叉验证特征选择完成！")
print("=" * 80)
print(f"关键产出:")
print(f"  1. 最终CatBoost模型: catboost_ncv_results['final_model']")
print(f"  2. 最终特征子集 ({len(final_feature_indices)}个): catboost_ncv_results['final_feature_indices']")
print(f"  3. 嵌套CV性能估计: AUC = {mean_outer_score:.4f} (±{std_outer_score:.4f})")
print(f"  4. 特征稳定性分析: catboost_ncv_results['feature_stability']")
print(f"\n下一步：使用 final_feature_indices 筛选测试集特征，并用 final_model 进行预测。")
