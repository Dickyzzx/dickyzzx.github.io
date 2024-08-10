---
title: 机器学习
date: 2024-08-03 00:12:10
---

# 前言
开这个page的是因为即将要去读研了，当作一次复习把之前学过的重新巩固一遍，所以对于简单概念依旧不会做太多解释，但是如果你也想学习机器学习可以根据我的内容进行查漏补缺。

## 机器学习相关术语
这里会随机添加想起来或遇到的术语，不需要仔细查看，但是用作概念复习是很好的办法。

    Model: An approximation of relationship between an input and output.

    Laplace Smoothing: A type of additive smoohting which mitigates the chance of encountering zero probabilities within the Naive Bayes classifier.

    Featurization: The process of transforming raw inputs into something a model can perform training and predictions on. 包括但不限于后续提到的0,1，2，3，

    Tokennization(0): The splitting of some raw textual input into individual words or elements.

    Stop word(1): A word, typically discarded, which doesn't add much predictive value, like this, is , a

    Stemming(2):Removing the ending modifiers of wards, leaving the stem of the word. studying ->study, studies -> studi

    Lemmatization(3): A more calculated form of stemming which ensures the proper lemma results from removing the word modifiers. studying->study, studies ->study ,but more expensive.

likelihood:似然率，其实算是不好理解的，但是[这篇文章](https://blog.csdn.net/jh1137921986/article/details/89000994)讲的好

## Supervise learning：监督学习
### 朴素贝叶斯

    核心理念:朴素贝叶斯分类器是基于贝叶斯定理和特征独立假设的简单而强大的概率分类器。贝叶斯定理给出了后验概率 𝑃(𝑦∣𝑋) 的计算公式，其中 X 是特征的合，y 是类别。不懂的请自行Google，有太多资源讲的比我好了。
    自我感觉难点在于
    联合概率P(a,b,c,d|y) = P(a|y)*...*P(d|y)
    后验概率P(y|a,b,c,d) = 
    ！！要计算后验证概率需要先计算联合概率。

    假设我们的y只有yes和no，a,b,c,d也是0 or 1
    我们要求P(y=yes|1 0 1 0) 和 P(y=no|1 0 1 0)
    一般来说过程如下

    1.先验概率：求P(y=yes) 和P(y=no)

    2.条件概率：分别求各个维度given yes 和given no的概率, 得到P(a|y)...P(d|y) 和P(a|n)...P(d|n)。

    3.似然函数（联合概率）：把条件概率按yes 和no 分类相乘。得到P(X|y=yes) 和P(X|y=no)

    4.证据（边际似然）： P(似然函数的yes) * P(先验概率的yes) + P(似然函数的no) * P(先验概率的no)，得到P(a,b,c,d)

    5.后验概率: 到这一步我们已经有了我们所需要的一切 直接套公式P(A|B)=....计算我们要求的即可。

    !!因为朴素贝叶斯假设特征之间是条件独立的 所P(d|y)*P(c|d,y)*P(b|c,d,y)*P(a|,b,c,d,y) = p(a|y)*p(b|y)*p(c|y)*p(d|y)
    
    为了避免0概率问题（分子为0）可以用Laplace smoothing。



### Performance
确定模型表现的一些方法：以检测信息是否spam为例。

    准确率(Accuracy):(TP+TN)/Total 但是并不总是一个好的指标。

    Sensitivity(recall): TP/(TP+FN) 可以理解为the model'ablity to correctly classfiy spam message.

    Specificity: TN/(TN+FP) represnts the classifier's ability to correctly classify legitiamte message. 

    更高Specificity就会拥有更少的假阳，更高的Sensitivity就会拥有更少的假阴，根据不同的情况会想要不一样的平衡。并不能一概而论。

    Precision: TP/(TP+FP),out of everytime, the model classified something as spam, how many of them actually were a spam. 越高越好

    F1-Score: 2*(Sensitivity*Precision)/(Sensitivity+Precission), The hormonic mean of the sensitivety and the precision.

关于 训练集，验证集，测试集请看[这里](https://blog.csdn.net/Swartz2015/article/details/78311592)



朴素贝叶斯优化

    Mutinomial Distribution : A distribution which models the probablity of counts of particular outcomes. 具体请搜索朴素贝叶斯和Mutinomial Distribution的例子。 除了多项式分布，还有二次项分布，高斯分布，等等建议都去了解一遍。

    TF-IDF:计算TF和IDF的值后相乘，具体如何计算请自行Google。具体来说，tf-idf是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。 字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。比如每个文章都有travel这个词，那这个词的意义就不大了。
    
    n-gram:详情访问https://blog.csdn.net/weixin_44966641/article/details/127479910

    



k-临近算法

各种距离的计算公式，特征规范，特征标准等
需要注意的是这个[kd树](https://blog.csdn.net/weixin_39910711/article/details/114447104)



决策树
    思考如何处理:missing value，multiple lable，regression。
    Classification and Regreesion Tree(CART):
        is an algorithm for constructing an approximate optimal decision tree for given example.
    
    Split point:
        A paire of feture and fetures value which is assigned to a node in a decision tree.
        This split point will determine which examples will go left and which examples go right based on the feature and feature value.
    
    Gini Impurity:
        Used as a way to determine the best split point for a given node in classification tree. It's based on the probability of incorrectly classifying an item based on all of the items in the node. 越低越好。 如果是用regression则是用mse测量。

    Surrogate Split:
        A suboptimal split point reserved for examples which are missing the optimal split point feature.
    
关于Boosting和Bagging看[这里](https://easyaitech.medium.com/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0-%E8%AF%A6%E8%A7%A3-bagging-boosting-%E4%BB%A5%E5%8F%8A%E4%BB%96%E4%BB%AC%E7%9A%84-4-%E7%82%B9%E5%8C%BA%E5%88%AB-6e3c72df05b8)


线性回归

    Variance Inflation Factor
        A measure of multicollinearity in a regreesion model.

    Feature Interaction
        Features that are multiplied by on another in order to express relationships that can't be represnted by adding the independent variable terms together.


逻辑回归

通俗理解，概率分布函数和概率密度函数：[链接一](https://cloud.tencent.com/developer/article/1514756)

在连续形变量中：

    概率分布函数(CDF)指的是累积概率，如某个人的身高小于或等于某个值的累积概率
    概率密度函数(PDF)指的是某个人的身高恰好在某个值附近的“可能性”。
    
    Softmax
        A sigmoid which is generalized to more than 2 classes to be predicted against.


支持向量机