### XGBoost 算法原理
XGBoost是一种提升树模型，将许多树（CART树）模型集成在一起，形成一个很强的分类器。

### XGBoost 损失函数

![image1](https://img-blog.csdn.net/20170228145205263)

### XGBoost 分裂结点算法
1. 暴力枚举：遍历所有特征的所有可能的分割点，计算Gain值，选取最大(Feature, label)去分裂
2. 近似方法：对于每个特征，只考察分位点，减少计算复杂度

### XGBoost 正则化
在XGBoost中，是将树的复杂度作为正则项加入，优化器在工作时会尽量不让这个树更复杂。

在给出n个实例，m维特征的情况下，
![image](https://image.jiqizhixin.com/uploads/editor/99a4523b-16e8-4549-a7e6-98c932cb69e1/image__1_.png)

q(x)代表一颗树，W代表叶子的分数，f（x）为样本实例的预测值，所以和CART的区别在于每个叶子节点有相应的权重Wi。为了学到模型需要的函数，需要定义正则化目标函数![image](https://image.jiqizhixin.com/uploads/editor/1e65f608-5978-44bb-b6a1-f54c2c09ae07/image__4_.png)

一种标准的正则化目标项= differentiable convex loss function + regularization，即损失函数+正则项。

L衡量预测值与真实值的差异，Ω作为模型复杂度的惩罚项，对于树的叶子节点个数和叶子节点权重的正则，防止过拟合，即simple is perfect，正则化项比RGF模型更加简单。


### XGBoost 对缺失值的处理
在XGBoost论文中关于缺失值的处理将其看作与稀疏矩阵的处理一样。在寻找split point的时候，不会对该特征为missing的样本进行遍历统计，只对该列特征值为non-missing的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找split point的时间开销。在逻辑实现上，为了保证完备性，会分别处理将missing该特征值的样本分配到左叶子结点和右叶子结点的两种情形，计算增益后选择增益大的方向进行分裂即可。可以为缺失值或者指定的值指定分支的默认方向，这能大大提升算法的效率。如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子树。


### XGBoost 优缺点
1. 优点 
- （1）不仅是CART树，还可以线性分类器 
- （2）引入正则化，提高模型的泛化能力 
- （3）基于预排序算法，并行训练 
- （4）对损失函数进行二阶泰勒展开，利用了一阶和二阶导数 
2. 缺点 
- （1）基于level-wise的分裂方式 
- （2）预排序方法空间消耗比较大，不仅要保存特征值，也要保存特征的排序索引，同时时间消耗也大


### XGBoost 应用场景
XGboost能够在一系列的问题上取得良好的效果，这些问题包括存销预测、物理事件分类、网页文本分类、顾客行为预测、点击率预测、动机探测、产品分类。多领域依赖数据分析和特征工程在这些结果中扮演重要的角色。XGBoost在所有场景中提供可扩展的功能，XGBoost可扩展性保证了相比其他系统更快速，XGBoost算法优势具体体现在：处理稀疏数据的新颖的树的学习算法、近似学习的分布式加权直方图。XGBoost能够基于外存的计算，保障了大数据的计算，使用少量的节点资源可处理大量的数据。XGBoost的主要贡献：

- 构建了高可扩展的端到端的boosting系统。
- 提出了具有合理理论支撑的分布分位调整框架。
- 介绍了一个新颖的并行适应稀疏处理树学习算法。
- 提出了基于缓存块的结构（cache-aware block structure）便于外存树（out-of-core tree）的学习。

### XGBoost的Sklearn参数

```
class xgboost.DMatrix(data, label=None, missing=None, weight=None, silent=False, feature_names=None, feature_types=None, nthread=None)
```
参数：
- data: DMatrix的数据源。当数据是字符串类型时，它表示路径libsvm格式txt文件，或者能xgboost读取的二进制文件
- label：训练数据的标签
- missing：需要以缺失值的形式表示的数据中的值
- weight：每个实例的权重
- silent：是否在构建期间打印信息
- feature_names：为特性设置名称
- feature_types：为特性设置类别
- nthread：从numpy从加载数据的线程数



#### 参考链接
https://www.cnblogs.com/wj-1314/p/9402324.html

https://blog.csdn.net/a819825294/article/details/51206410

https://zhuanlan.zhihu.com/p/38297689

https://www.jiqizhixin.com/graph/technologies/ea0eb940-c873-42bc-a752-0a07f15a52c0
