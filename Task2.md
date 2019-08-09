### 1. 前向分步算法
给定训练数据集：
```math
T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}, x_i \in X \subseteq R^n, y_i \in Y=\{-1,+1\}
```
损失函数 L(y,f(x))，基函数的集合{b(x;γ)}；学习加法模型的前向分步的算法如下：
1. 初始化f0(x) = 0
2. 对m=1,2,3,...,M （M为基函数的个数）
-  （a）在上一步得到的最优基函数的基础上，极小化本次单个基函数的损失函数，得到本轮最优参数β_m,γ_m
```math
(β_m,γ_m) = argmin_{β,γ}\sum_{i=1}^NL(y_i,f_{m-1}(x_i) + βb(x_i;γ))
```
-   (b)更新线性累加基函数
```math
f_m(x) = f_{m-1}(x)+β_mb(x;γ_m)
```
3. 得到最终加法模型
```math
f(x) = f_M(x) = \sum_{m=1}^Mβ_mb(x;γ_m)
```
这样，前向分步算法将同时求解m=1到M的所有参数β_m,γ_m的全局最优解问题++简化为++**逐次**求解各个β_m,γ_m的局部最优化问题。


### 2. 负梯度拟合
Freidman提出用损失函数的负梯度来拟合本轮损失的近似值，进而拟合一个CART回归树。第t轮的第i个样本的损失函数的负梯度表示为
```math
r_{ti} = -\bigg[\frac{\partial L(y_i, f(x_i)))}{\partial f(x_i)}\bigg]_{f(x) = f_{t-1}\;\; (x)}
```
利用(xi, rti) (i = 1,2,...,m)，我们可以拟合一颗CART回归树，得到了第t棵回归树，其对应的叶节点区域Rtj，j = 1,2,...J。其中J为叶子节点的个数。

针对每一个叶子节点里的样本，我们求出使损失函数最小，即拟合叶子节点最好的输出值ctj
```math
c_{tj} = \underbrace{arg\; min}_{c}\sum\limits_{x_i \in R_{tj}} L(y_i,f_{t-1}(x_i) +c)
```
这样就得到本轮的决策树拟合函数
```math
h_t(x) = \sum\limits_{j=1}^{J}c_{tj}I(x \in R_{tj})
```
从而本轮最终得到的强学习器表达式
```math
f_{t}(x) = f_{t-1}(x) + \sum\limits_{j=1}^{J}c_{tj}I(x \in R_{tj})
```
通过损失函数的负梯度来拟合这种通用的拟合损失误差的办法，就可用GBDT解决分类回归问题，区别仅在损失函数不同导致的负梯度不同。


### 3. 损失函数
提升树算法中的损失函数是
```math
L(y,f_{t-1}(x)) = L(y,f_{t-1}(x)+h_t(x))
```
当采用平方损失函数时
```math
L(y,f_{t-1}(x)+h_t(x)) = (y-f_{t-1}(x)-h_t(x))^2 = (r-h_t(x))^2
```

### 4. 回归
输入：训练集样本T={(x_1,y_1),(x_2,y_2),...,(x_m,y_m)}，最大迭代次数T，损失函数L
输出：强学习器f(x)
1. 初始化弱学习器
```math
f_0(x) = \underbrace{arg\; min}_{c}\sum\limits_{i=1}^{m}L(y_i, c)
```
2. 对迭代轮数t=1,2,...,T
- a) 对样本i=1,2,...,m,计算负梯度
```math
r_{ti} = -\bigg[\frac{\partial L(y_i, f(x_i)))}{\partial f(x_i)}\bigg]_{f(x) = f_{t-1}\;\; (x)}
```
- b) 利用(xi,rti) (i=1,2,...,m)，拟合一棵CART回归树，得到第t棵回归树，其对应的叶子节点区域Rtj,j=1,2,...,J。J为回归树t的叶子节点的个数
- c) 对叶子区域j =1,2,..J,计算最佳拟合值
```math
c_{tj} = \underbrace{arg\; min}_{c}\sum\limits_{x_i \in R_{tj}} L(y_i,f_{t-1}(x_i) +c)
```
- d) 更新强学习器
```math
f_{t}(x) = f_{t-1}(x) + \sum\limits_{j=1}^{J}c_{tj}I(x \in R_{tj})
```

3. 得到强学习器f(x)的表达式
```math
f(x) = f_T(x) =f_0(x) + \sum\limits_{t=1}^{T}\sum\limits_{j=1}^{J}c_{tj}I(x \in R_{tj})
```


### 5. 分类算法
GBDT的分类和回归算法从思想上没有区别，但由于样本输出不是连续的值，而是离散的类别，使我们无法直接从输出类别去拟合类别输出的误差。
#### 5.1  二分类
若用类似逻辑回归的对数似然损失函数，则损失函数为
```math
L(y, f(x)) = log(1+ exp(-yf(x)))
```
此时负梯度误差为
```math
r_{ti} = -\bigg[\frac{\partial L(y, f(x_i)))}{\partial f(x_i)}\bigg]_{f(x) = f_{t-1}\;\; (x)} = \frac{y_i}{(1+exp(y_if(x_i)))}
```
对于生成的决策树，我们各个叶子节点的最佳负梯度拟合值为
```math
c_{tj} = \underbrace{arg\; min}_{c}\sum\limits_{x_i \in R_{tj}} log(1+exp(-y_i(f_{t-1}(x_i) +c)))
```
由于上式比较难优化，我们一般使用近似值代替
```math
c_{tj} = \frac{\sum\limits_{x_i \in R_{tj}}r_{ti}}{\sum\limits_{x_i \in R_{tj}}|r_{ti}|(1-|r_{ti}|)}
```


#### 5.2  多分类
假设类别数为K，此时的对数似然损失函数为：
```math
L(y, f(x)) = - \sum\limits_{k=1}^{K}y_klogp_k(x)
```
第k类的概率表达式为：
```math
p_k(x) = \frac{exp(f_k(x))}{\sum\limits_{l=1}^{K} exp(f_l(x))}
```
结合上面两个式子，算出第t轮的第i个样本对应类别l的负梯度误差为
```math
r_{til} = -\bigg[\frac{\partial L(y_i, f(x_i)))}{\partial f(x_i)}\bigg]_{f_k(x) = f_{l, t-1}\;\; (x)} = y_{il} - p_{l, t-1}(x_i)
```
观察上式可以看出，其实这里的误差就是样本𝑖对应类别𝑙的真实概率和𝑡−1轮预测概率的差值。

对于生成的决策树，我们各个叶子节点的最佳负梯度拟合值为
```math
c_{tjl} = \underbrace{arg\; min}_{c_{jl}}\sum\limits_{i=0}^{m}\sum\limits_{k=1}^{K} L(y_k, f_{t-1, l}(x) + \sum\limits_{j=0}^{J}c_{jl} I(x_i \in R_{tjl}))
```
由于上式比较难优化，我们一般使用近似值代替
```math
c_{tjl} =  \frac{K-1}{K} \; \frac{\sum\limits_{x_i \in R_{tjl}}r_{til}}{\sum\limits_{x_i \in R_{til}}|r_{til}|(1-|r_{til}|)}
```
除了负梯度计算和叶子节点的最佳负梯度拟合的线性搜索，多元GBDT分类和二元GBDT分类以及GBDT回归算法过程相同。


### 6. 正则化
GBDT的正则化主要有三种方式
1. 和Adaboost类似的正则化项，即步长。
2. 通过子采样（不放回抽样）
3. 对于弱学习器即CART回归树进行正则化剪枝。


### 7. 优缺点
GBDT的优点有：
- 可以灵活处理各种类型的数据，包括连续值和离散值
- 在相对少的调参时间下，预测的准确率（相对SVM来说）也可较高
- 使用一些健壮的损失函数，对异常值的鲁棒性很强（如Huber损失函数和Quantile损失函数）

GBDT的缺点有：
- 由于弱学习器之间存在依赖关系，难以并行训练数据。但可以通过自采样的SGBT达到部分并行


### 8. sklearn参数
```
class sklearn.ensemble.GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
```
参数：
- 损失 ： {'deviance'，'exponential'}，可选（默认='deviance'）。损失函数要优化。“偏差”是指用概率输出进行分类的偏差（=逻辑回归）。对于损失'指数'梯度增强恢复AdaBoost算法。
- learning_rate ： float，optional（默认值= 0.1）。学习率会缩小每棵树的贡献learning_rate。在learning_rate和n_estimators之间进行权衡。
- n_estimators ： int（默认值= 100）。要执行的助推阶段的数量。对于过度拟合，梯度增强相当稳健，因此大量通常会产生更好的性能。
- subsample ： float，optional（默认值= 1.0）。用于拟合各个基础学习者的样本分数。如果小于1.0，则会导致随机梯度提升。subsample与参数交互n_estimators。选择导致减少差异和增加偏差。subsample < 1.0
- criterion ： string，optional（default =“friedman_mse”）。衡量分裂质量的功能。支持的标准是弗里德曼的改进得分的均方误差“friedman_mse”，均方误差的“mse”和平均绝对误差的“mae”。默认值“friedman_mse”通常是最好的，因为它可以在某些情况下提供更好的近似值。
- min_samples_split ： int，float，optional（default = 2）。拆分内部节点所需的最小样本数
- min_samples_leaf ： int，float，optional（default = 1）。叶节点所需的最小样本数。
- min_weight_fraction_leaf ： float，optional（默认= 0）。需要在叶节点处的权重总和（所有输入样本的总和）的最小加权分数。当未提供sample_weight时，样本具有相同的权重。
- max_depth ： 整数，可选（默认= 3）。个体回归估计量的最大深度。最大深度限制树中的节点数。调整此参数以获得最佳性能; 最佳值取决于输入变量的相互作用。
- min_impurity_decrease ： float，optional（默认= 0。）
- 如果该分裂导致杂质的减少大于或等于该值，则将分裂节点。
- min_impurity_split ： float，（默认值= 1e-7）
- 树木生长早期停止的门槛。如果节点的杂质高于阈值，节点将分裂，否则它是叶子。
- 
- init ： estimator或'zero'，可选（默认=无）。用于计算初始预测的估算器对象。 init必须提供fit和predict_proba。如果为“零”，则初始原始预测设置为零。默认情况下，使用 DummyEstimator预测类先验。
- random_state ： int，RandomState实例或None，可选（默认=无）
- max_features ： int，float，string或None，可选（默认=无）。寻找最佳分割时要考虑的功能数量
- verbose ： int，默认值：0 。启用详细输出。如果为1则会偶尔打印进度和性能（树越多，频率越低）。如果大于1，则它会打印每棵树的进度和性能。
- max_leaf_nodes ： int或None，可选（默认=无）
- max_leaf_nodes以最好的方式种植树木。最佳节点定义为杂质的相对减少。如果为None则无限数量的叶节点。
- warm_start ： bool，默认值：False。设置True为时，重用上一个调用的解决方案以适合并向集合中添加更多估计器，否则，只需擦除以前的解决方案。请参阅词汇表。
- presort ： bool或'auto'，可选（默认='自动'）。是否预先分配数据以加快拟合中最佳分割的发现。默认情况下，自动模式将使用密集数据上的预分类，并默认对稀疏数据进行常规排序。在稀疏数据上将presort设置为true将引发错误。
- validation_fraction ： float，optional，默认值为0.1。将训练数据的比例留作早期停止的验证集。必须介于0和1之间。仅在n_iter_no_change设置为整数时使用。
- n_iter_no_change ： int，默认不训练。默认情况下，它设置为“None”以禁用提前停止。如果设置为数字，它将保留validation_fraction训练数据的大小作为验证，并在验证得分在所有先前n_iter_no_change的迭代次数中没有改善时终止训练。分裂是分层的。
- tol ： float，optional，默认值1e-4。允许在早期停止。如果损失没有至少改善tol的n_iter_no_change迭代次数（如果设置为数字），则训练停止。


### 9. 应用场景
- 推荐场景
- 搜索引擎