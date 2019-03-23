from sklearn import tree                                 #导入sklearn模块
from sklearn.datasets import load_wine                   #使用sklearn自带数据集
from sklearn.model_selection import train_test_split     #使用自带的包划分测试集和训练集
import graphviz
import pandas as pd

wine = load_wine()

pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)   #将红酒的数据和特征做成表格的形式
wine.feature_names
wine.target_names
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
Xtrain.shape
wine.data.shape
clf = tree.DecisionTreeClassifier(criterion='entropy'
                                  ,random_state=30
                                  ,splitter='random')
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print(score)


feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']

dot_data = tree.export_graphviz(clf
                                ,feature_names = feature_name
                                ,class_names = ['琴酒','雪莉','贝尔摩德']
                                ,filled=True
                                ,rounded=True)
graph = graphviz.Source(dot_data)


clf.feature_importances_    #特征的重要性排名
#[*zip(feature_name,clf.feature_importance_)]              #将重要性和特征名对应起来
