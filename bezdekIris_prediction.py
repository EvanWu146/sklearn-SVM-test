# -*- coding:utf-8 -*-
from sklearn import svm, tree
import graphviz

train_X = []
train_Y = []
test_class = []
pos_class = 0
neg_class = 0
f = open('bezdekIris.txt', "r")
bezdekIris_class = ['Iris-versicolor', 'Iris-setosa']

for line in f:
    if len(line.strip().split(",")) == 5:
        x1, x2, x3, x4, cate = line.strip().split(",")
        if cate == "Iris-setosa":
            if pos_class < 40:  # 取前四十个做训练集合
                train_X.append([x1, x2, x3, x4])
                train_Y.append(1)  # 标记为正类
                pos_class += 1
            else:
                test_class.append([x1, x2, x3, x4])
        elif cate == 'Iris-versicolor':
            if neg_class < 40:
                train_X.append([x1, x2, x3, x4])
                train_Y.append(0)  # 标记为负类
                neg_class += 1
            else:
                test_class.append([x1, x2, x3, x4])

f.close()

#--------线性核支持向量训练及测试----------------------------------
clf0 = svm.SVC(kernel='linear')
clf0.fit(train_X, train_Y)
print("线性核函数的支持向量为：\n", clf0.support_vectors_)
print("每个类别的支持向量的个数：\n", clf0.n_support_)
for vector in test_class:
    tem = clf0.predict([vector])
    print("\n测试向量为："+str(vector)+'\n'+'测试结果为：'+str(tem))
    print('属于'+bezdekIris_class[tem[0]])

#--------高斯核支持向量训练及测试----------------------------------
clf1 = svm.SVC(kernel='rbf')
clf1.fit(train_X, train_Y)
print("\n高斯核函数的支持向量为：\n", clf1.support_vectors_)
print("每个类别的支持向量的个数：\n", clf1.n_support_)
for vector in test_class:
    tem = clf1.predict([vector])
    print("\n测试向量为：" + str(vector) + '\n' + '测试结果为：' + str(tem))
    print('属于' + bezdekIris_class[tem[0]])

#--------决策树训练及测试----------------------------------
clf2 = tree.DecisionTreeClassifier()
clf2.fit(train_X, train_Y)
print("\n决策树训练及测试")
for vector in test_class:
    tem = clf2.predict([vector])
    print("\n测试向量为：" + str(vector) + '\n' + '测试结果为：' + str(tem))
    print('属于' + bezdekIris_class[tem[0]])
