# -*- coding:utf-8 -*-
import csv
from sklearn import svm,tree


train_X = []
train_Y = []
test_class = []
pos_class = 0
neg_class = 0
test_num = 0
class_id = ['M', 'L']

with open('ai4i2020.csv') as f:
    reader = csv.reader(f)  # 返回一个reader的迭代器
    head_row = next(reader)  # next获取标题行
    for line in reader:
        tem_X = line[3:7]  # 只取3、4、5、6行组成训练向量
        tem_Y = line[2]  # 类别标签

        if tem_Y == 'L' and pos_class < 40:
            train_X.append(tem_X)
            train_Y.append(1)
            pos_class += 1
        elif tem_Y == 'M' and neg_class < 40:
            train_X.append(tem_X)
            train_Y.append(0)
            neg_class += 1

        if pos_class == 40 and neg_class == 40:
            if test_num < 10:
                test_class.append(tem_X)
                test_num += 1
            else:
                break

    f.close()
    print(train_X)
    print(train_Y)
    print(test_class)


#--------线性核支持向量训练及测试----------------------------------
clf0 = svm.SVC(kernel='linear')
clf0.fit(train_X, train_Y)
print("线性核函数的支持向量为：\n", clf0.support_vectors_)
print("每个类别的支持向量的个数：\n", clf0.n_support_)
for vector in test_class:
    tem = clf0.predict([vector])
    print("\n测试向量为："+str(vector)+'\n'+'测试结果为：'+str(tem))
    print('属于'+class_id[tem[0]])

#--------高斯核支持向量训练及测试----------------------------------
clf1 = svm.SVC(kernel='rbf')
clf1.fit(train_X, train_Y)
print("\n高斯核函数的支持向量为：\n", clf1.support_vectors_)
print("每个类别的支持向量的个数：\n", clf1.n_support_)
for vector in test_class:
    tem = clf1.predict([vector])
    print("\n测试向量为：" + str(vector) + '\n' + '测试结果为：' + str(tem))
    print('属于' + class_id[tem[0]])

#--------决策树训练及测试----------------------------------
clf2 = tree.DecisionTreeClassifier()
clf2.fit(train_X, train_Y)
print("\n决策树训练及测试")
for vector in test_class:
    tem = clf2.predict([vector])
    print("\n测试向量为：" + str(vector) + '\n' + '测试结果为：' + str(tem))
    print('属于' + class_id[tem[0]])
