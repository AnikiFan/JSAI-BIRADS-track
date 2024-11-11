import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.svm import SVC
import optuna
import warnings
from joblib import dump, load
warnings.filterwarnings('ignore')

class fea2cla_model:
    def __init__(self,basic_features,map_label, best_model=None, random_state=42,enhance=True):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        self.feature_columns = None
        self.is_fitted = False
        self.n_trials = None # 搜索次数 
        self.basic_features = basic_features # 基础特征
        self.map_label = map_label # 划分类别
        self.enhance = enhance # 是否增强特征
        self.classification_report = None # 模型评估结果
    def create_advanced_features(self, X, basic_features):
        """
        创建高级特征
        
        参数:
            data: DataFrame, 原始数据
            basic_features: list, 基础特征列表
        """
        features = X[basic_features].copy()
        
        # 特征交互
        for i, feat1 in enumerate(basic_features):
            for j, feat2 in enumerate(basic_features):
                if i < j:
                    features[f'{feat1}_{feat2}_mul'] = features[feat1] * features[feat2]
                    features[f'{feat1}_{feat2}_div'] = features[feat1] / (features[feat2] + 1e-6)
        
        # 统计特征
        features['mean'] = features[basic_features].mean(axis=1)
        features['std'] = features[basic_features].std(axis=1)
        features['max'] = features[basic_features].max(axis=1)
        features['min'] = features[basic_features].min(axis=1)
        features['sum'] = features[basic_features].sum(axis=1)
        
        # 多项式特征
        for feat in basic_features:
            features[f'{feat}_squared'] = features[feat] ** 2
            features[f'{feat}_cubed'] = features[feat] ** 3
        
        return features
    
    def optimize_xgboost(self, trial, X_train, y_train):
        param = {
        # 基础参数
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        
        # 防止过拟合参数
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        
        # 类别不平衡相关参数  
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0), # worse
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10), # worse
        
        # # 正则化参数
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        
        # 其他参数
        'random_state': self.random_state,
        'tree_method': 'hist',  # 更快的训练速度
        }
        
        model = xgb.XGBClassifier(**param)
        
        # 使用更少的折数
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=3,  # 减少折数
            scoring=lambda estimator, X, y: (
                0.6 * accuracy_score(y, estimator.predict(X)) + 
                0.4 * f1_score(y, estimator.predict(X), average='macro')
            )
        )
        
        return cv_scores.mean()
    
    def train_and_evaluate(self,data,test_size=0.2, n_trials=20):
        self.n_trials = n_trials
        # X
        X = data[self.basic_features].copy()
        assert X.isnull().sum().sum() == 0, "数据中存在缺失值"
        if self.enhance:
            X = self.create_advanced_features(X, self.basic_features)  
        X_scaled = self.scaler.fit_transform(X) # 标准化
        # y
        y = data['label'].map(self.map_label)
        assert y.isnull().sum() == 0, "标签中存在缺失值"
        self.feature_columns = X.columns.tolist()
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.optimize_xgboost(trial, X_train, y_train), 
                      n_trials=n_trials)
        
        best_params = study.best_params
        self.best_model = xgb.XGBClassifier(**best_params, random_state=42)
        self.best_model.fit(X_train, y_train)
        
        pred = self.best_model.predict(X_test)
        self.classification_report = classification_report(y_test, pred)
        self.is_fitted = True
        self.feature_importance = pd.DataFrame({
            '特征': X.columns,
            '重要性': self.best_model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        self.show_classification_report()
        return self
    def show_classification_report(self):
        print("="*100)
        print("划分类别：", self.map_label)
        print("增强特征：", self.enhance)
        print("特征：", self.feature_columns)
        print("搜索次数：", self.n_trials)
        print("模型评估结果：", self.classification_report)
        print("特征重要性：", self.feature_importance)
        print("="*100)
        
    def predict(self, X, return_prob=False):
        """使用训练好的模型进行预测"""
        # check X 有 basic_features 列
        try:
            X = X[self.basic_features]
        except KeyError:
            raise ValueError(f"X 中缺少基础特征: {set(self.basic_features) - set(X.columns)}")
        if self.enhance:
            print("增强特征：", self.enhance)
            X = self.create_advanced_features(X, self.basic_features)
        # print("特征：", X.columns)
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train_and_evaluate方法")
        
        X_scaled = self.scaler.transform(X)
        if return_prob:
            return self.best_model.predict_proba(X_scaled)
        else:
            return self.best_model.predict(X_scaled)
    
    def save_model(self, filepath):
        """保存模型和相关组件"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted,
            'basic_features': self.basic_features,
            'map_label': self.map_label,
            'enhance': self.enhance,
            'classification_report': self.classification_report,
            'n_trials': self.n_trials,
            'feature_columns': self.feature_columns
        }
        dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """加载保存的模型和相关组件"""
        model_data = load(filepath)
        
        # 使用保存的basic_features和map_label创建实例
        trainer = cls(
            basic_features=model_data['basic_features'],
            map_label=model_data['map_label']
        )
        
        trainer.best_model = model_data['model']
        trainer.scaler = model_data['scaler']
        trainer.feature_importance = model_data['feature_importance']
        trainer.is_fitted = model_data['is_fitted']
        trainer.enhance = model_data['enhance']
        trainer.classification_report = model_data['classification_report']
        trainer.n_trials = model_data['n_trials']
        trainer.feature_columns = model_data['feature_columns']
        
        return trainer