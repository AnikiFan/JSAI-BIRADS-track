import cv2
from sklearn.preprocessing import StandardScaler
from joblib import  load
def cv_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转换为单通道
    _,thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY) # 设置阈值
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 轮廓检测
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea)) # 提取出最大的轮廓
    return x,y,w,h

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