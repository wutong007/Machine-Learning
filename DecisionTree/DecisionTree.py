from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

wine = load_wine()
pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
wine.feature_names
wine.target_names
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
print('训练集的容量为：', Xtrain.shape)
print('样本的容量为：', wine.data.shape)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
print('预测的精度为：', score)

feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']

dot_data = tree.export_graphviz(clf
                                , feature_names=feature_name
                                , class_names=['琴酒', '雪莉', '贝尔摩德']
                                , filled=True
                                , rounded=True)
graph = graphviz.Source(dot_data)
graph.render("tree.png")
