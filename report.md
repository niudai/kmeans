# 大数据算法报告: K-means 算法

### 摘要

K-means 是常用的基于欧式距离的聚类算法，其分类的依据是两个目标距离越近，相似度就越高。聚类的目的是找到几个聚类中心，让所有的数据点到这些聚类中心的距离之和达到最小。虽然 K-means 算法本质是一个局部搜索算法，但是其算法的迭代性和快速收敛性使得它仍然是一种被广泛采用的聚类算法。本文介绍 K-means 基本算法的原理，并再此基础上增加 K-means++ 的证明过程。并且对于有噪音的样本数据点，提出一些具体的解决方案。

### 引言

K-means 算法是一种迭代算法，它试图将数据集划分为 K 个不同的聚类，每个数据点属于一类，算法过程如下：

1. 指定聚类的个数 K。
2. 随机选取样本点中的 K 个，作为初始聚类中心点。
3. 把每个数据点分配给最近的聚类中心。
4. 中心用质心的方法重新定位每类的中心。
5. 重复 3-4，直至聚类中心位置稳定。

本质上来说，算法的目的是为了将目标函数降至最低，目标函数即为每个点距离聚类中心距离的平方之和：

$$\sum_{1}^{m} ||x^i - c^i||^2 $$

$c^i$ 是 $x^i$ 所属的聚类中心点。

上述的算法，时间复杂度为 $O(kmt)$, 其中 t 为迭代次数，k 为分类的类数，m 为样本个数。

由于只需为每个样本点和聚类中心保存位置，所以空间复杂度为 $O(m+k)$。

基于此，K-means 算法总能快速地收敛，找到局部最优解，但是问题是局部最优不一定是全局最优，比如 K-means 可能做出如下分类：

![Image](https://pic4.zhimg.com/80/v2-3a2faa8e13d73fd001ebbc1062691add.png)

![Image](https://pic4.zhimg.com/80/v2-b0a11c1b83cfecb56161e81f5121e2ce.png)

可以看到第一个分类较合理，第二个分类不合理，所以以往的方法就是通过多次随机初始化，最好找那个距离和最小的分类。

*K-means++* 算法对此做了改进，它在聚类中心的初始化方面，使用距离加权概率的方式进行选择，能以更大的概率逼近全局最优，从而避免多次使用 *K-means* 算法，而且因为本身的聚类中心选择策略也导致了它能更快的收敛至稳定状态，提升了算法的性能。

## *K-means++* 算法

*K-means++* 算法只在最开始的聚类中心点初始化和原版的不同，因此只介绍初始化的步骤：

1. 从样本点中随机选取一点 $c1$ 作为第一个聚类中心。
2. 计算每个点到最近的聚类中心的距离的平方 $D(x_i)$, 然后按照概率 $\frac{D(x_i)^2}{\sum_{D(x)\in X}D(x)^2}$ 选取一个点作为下一个聚类中心。
3. 重复上述步骤直到选满 k 个聚类中心。
4. 运行 *K-means* 算法。

## 证明

>THEOREM I: 如果分类 $C$ 是由 k-means++ 算法计算得出的，那么势能函数 $\phi$ 满足 $E[\phi]\leq8(lnk + 2)\phi_{opt}$

上述定理中，$\phi_{OPT}$ 是指理论最优分类对应的势能函数。

求解理论最优分类是一个 NP-Complete 问题，理论上无法通过多项式时间内获得解，但是 *K-means++* 保证了和最优解之间的接近性。

>LEMMA 2.1 令 S 是一系列点，$c(S)$ 是对应的质心，令 $z$ 是任意一点，那么有 $\sum_{x\in S}||x-z||^2 - \sum_{x\in S}||x-c(S)||^2=|S|\times||c(S)-Z||^2$

>LEMMA 3.1 令 A 是 $C_{opt}$ 的任意一个聚类，令 $C$ 是任意一个只有一个中心的分类，这个中心是从 A 中均匀分布随机选取的，那么有 $E[\phi(A)]=2\phi_{opt}(A)$.

证明：

令 $c(A)$ 表示聚类 A 的质心，所以 $E[\phi(A)]$ 可以下式给出：

$$
    \sum_{a_0\in A}\frac{1}{|A|} \times (\sum_{a\in A}||a-a_0||^2)
$$

>LEMMA 3.2 令 A 是 $C_{opt}$ 中任意的一个聚类，令 $C$ 是任意的分类。如果从 A 中添加一个随机的质心到 $C$，选择的概率按照 $D^2$ 作为概率权重，那么有 $E[\phi(A)] \leq 8\phi_{opt}(A)$ 。

证明：

由上面可以直到，我们从 A中 $a_0$ 作为中心的概率为 $\frac{D{a_0}^2}{\sum_{a\in A}D(a)^2}$。所以有：

$$
	E[\phi(A)] = \sum_{a_0\in A}\frac{D(a_0)^2}{\sum_{a \in A}D(a)^2} \times \sum_{a \in A} min(D(a), ||a-a_0||^2)
$$

由三角不等式可得：

$$
 D(a_0) \leq D(a) + ||a-a_0||
$$

由此进一步可得：

$$
	D(a_0)^2 \leq 2D(a)^2 + 2||a-a_0||^2
$$

关于所有的 a 求和，可以得到：

$$
	D(a_0)^2 \leq \frac{2}{|A|}\sum_{a \in A}D(a)^2 + \frac{2}{|A|}\sum_{a \in A}||a-a_0||^2 
$$	

第一个表达式里，我们由 $min(D(a), ||a-a_0||)^2 \leq ||a-a_0||^2$ 做替换，在第二个表达式中，我们做 $min(D(a), ||a-a_0||)^2 \leq D(a)^2$。简化后为：

$$
	\frac{2}{|A|}\times\sum_{a_0\in A} \frac{\sum_{a \in A} D(a)^2}{\sum_{a \in A} D(a)^2}\times \sum_{a \in A}min(D(a), ||a-a_0||^2) + \newline \frac{2}{|A|}\times\sum_{a_0 \in A}\frac{\sum_{a \in A}||a-a_0||^2}{\sum_{a \in A}D(a)^2}\times\sum_{a \in A}min(D(a), ||a-a_0||^2)

$$

$$
	E[\phi(A)] \leq \frac{4}{|A|}\times\sum_{a_0 \in A}\sum_{a \in A}||a-a_0||^2 = 8\times\phi_{opt}(A)
$$

我们可以看到由 $D^2$ 进行概率权重的选取是 competivie 的，我们由此证明最大的误差是 $O(logk)$ 的。

我们现在证明了从最优分类里按照距离平方概率权重选取分类中心，可以让得到的新分类有：

$$
    E[\phi(A)] \leq 8\phi_{opt}(A)
$$

> Lemmma 3.3 令 $C$ 是任意一个分类，从 $C_{opt}$ 中选取 u 个未被覆盖的集群，并令 $X_u$ 表示这些聚类中的点。并令 $X_c = X - X_u$, 也就是说，$X_c$ 是已经覆盖的聚类里的点，现在假定我们添加 $t \leq u$ 个随机中心到 $C$，选择 $D^2$ 作为概率权重，令 $C^`$ 表示新的聚类，令 $\phi^`$ 是对应的势，那么 $E[\phi^`]$ 最多为：

$$ (\phi(X_c) + 8\phi_{opt}(X_u))\times(1+H_t)+\frac{u-t}{u}\times\phi(X_u)
$$ 

其中 $H_t$ 代表调和求和：$1+\frac12+...+\frac1t$

证明：

通过数学归纳法证明。只需证明如果 $(t-1, u)$ 和 $(t-1, u-1)$，那么它也满足 $(t, u)$.

如果 $t = 0$, 并且 $u > 0$, 那么就有 $1 + H_t = \frac{u-t}{u} = 1$, 下一步假如 $t = u = 1$. 我们从未被覆盖的聚类中按照概率 $\frac{\phi(X_u)}{\phi}$. 那么有：

$$
	E[\phi] \leq \frac{\phi(X_u)}{\phi}\times(\phi(X_c) + 8\phi_{opt}(X_u)) + \newline \frac{\phi(X_c)}{\phi}\times{\phi} \leq 2\phi(X_c) + 8\phi_{opt}(X_u)
$$

因为 $1 + H_t=2$, 所以当 $t=u=1$时是成立的。

接下来只需证明递推的部分。
考虑两种情况，首先假定我们从覆盖聚类中选取一个中心，这个新中心做多降低 $\phi$。接下来又 $E[\phi]$
 的贡献量最多：

$$
	\frac{\phi(X_c)}{\phi}\times((\phi(X_c)+8\phi_{opt}(X_u))\times(1+H_(t-1))+\frac{u-t+1}{u}\times\phi(X_u))
$$

另一方面，假如我们从未覆盖的聚类中选取一个中心，令 $p_a$ 作为我们选取 $a\in A$ 作为中心的概率。让 $\phi_a$ 表示选择 $a$ 作为中心后的分类。然后令 t 和 u 减一，那么 $E[\phi_{opt}]$ 最多是：

$$
	\frac{\phi(A)}{\phi}\times\sum_{a\in A}p_a((\phi(X_c)+\phi_a+8\phi_{opt}(X_u)-8\phi_{opt}(A))\times(1+H_{t-1})+\newline \frac{u-t}{u-1}\times(\phi(X_u)-\phi(A))) \leq \frac{\phi(A)}{\phi}\times((\phi(X_c)+8\times\phi_{opt}(X_u))\times(1+H_{t-1})+\newline \frac{u-t}{u-1}\times(]phi(X_u)-\phi(A)))
$$

根据平方平均不等式，可以得到 $\sum_{a\in X_u}\phi(A)^2 \leq \frac{1}{u
}\times\phi(X_u)^2$, 所以关于所有未覆盖的聚类，就和，可以得到势的贡献最多是：

$$
\frac{\phi(X_u)}{\phi}\times(\phi(X_c)+8\phi_{opt}(X_u))\times(1+H_{t-1})+\frac{1}{\phi}\times\frac{u-t}{u-1}\times(\phi(X_u)^2 - \frac{1}{u}\times\phi(X_u)^2) = \newline \phi(X_u)\phi\times((\phi(X_c)+8\phi_{opt}(X_u))\times(1+H_{t-1})+\frac{u-t}{u}\times\phi(X_u))
$$

于是：

$$
	E[\phi] \leq (\phi(X_c)+8\phi_{opt}(X_u))\times(1+H_{t-1})+ \newline \frac{u-t}{u}\times\phi(X_u)+\frac{\phi(X_c)}{\phi}\times\frac{\phi(X_u)}{u} \leq (\phi(X_c) + 8\phi_{opt}(X_u)) \newline \times(1+H_{t-1}+\frac{1}{u})+\frac{u-t}{u}\phi(X_u)
$$

> Theorem 3.1 如果分类 $C$ 是由 *k-means++* 构建的，那么对应的势能函数 $E[\phi] \leq 8(lnk + 2)\times\phi_{opt}$

证明：

考虑完成了算法第一步的分类，让 A 表示 $C_{opt}$ 聚类，应用 *Lemma 3.3* 有 $t=u=k-1$ 并且 A 作为唯一的覆盖后的集群，我们有：

$$
	E[\phi_{opt}]\leq(\phi(A)+8\phi_{opt}-8\phi_{opt}(A))\times(1+H_{k-1})
$$

并且对于调和级数，有：

$$
    H_{k-1} \leq (1 + lnk)
$$

由此得：

$$
    E[\phi] \leq (\phi(A) + 8\phi_{opt} - 8\phi_{opt}(A))(2 + lnk) \leq 8\phi_{opt}(1+H_{k-1})
$$

## 使用 K-means-- 方法处理带离群点的问题

### 引言 

对于一系列样本点，有可能会出现离群点，这些点可能会影响我们的分类过程，所以要建立一种检测，筛选离群点的机制。

在论文 [k-means--: A unified approach to clustering and outlier detection](http://users.ics.aalto.fi/gionis/kmmm.pdf) 一文中，阐述了一种能够检测离群点的算法，算法的核心思想是，每次更新聚类中心的时候，将最远的 l 个排除。

算法的过程如下：

```
输入: 一系列样本点 X = {x1, ..., xn}
距离函数 d : X × X → R
分类的数量 k 和离群点个数 l
输出: k 个集合，聚类中心为 C
A set of ` outliers L ⊆ X
C_0 ← {k random points of X}
i ← 1
while (no convergence achieved) do
Compute d(x | Ci−1), for all x ∈ X
Re-order the points in X such that
d(x1 | Ci−1) ≥ . . . ≥ d(xn | Ci−1)
Li ← {x1, . . . , x`}
Xi ← X \ Li = {x`+1, . . . , xn}
for (j ∈ {1, . . . , k}) do
Pj ← {x ∈ Xi
| c(x | Ci−1) = ci−1,j}
ci,j ← mean(Pj )
Ci ← {ci,1, . . . , ci,k}
i ← i + 1
```

可以看到和 *K-means* 算法的核心区别是每次计算样本点和最近聚类中心的距离后，会根据距离进行排序，将距离最大的 l 个排除考虑，只根据 m-l 个样本点计算新的聚类中心，这样每次更新可以把距离最远的 l 个忽略掉，从而某种意义上消弱了离群点对分类的影响。

## 算法实现

### K-means

在该报告中，分别实现了平凡的 *K-means* 算法和本文所讲的 *K-means++* 算法，并且针对有带离群点的 k-means 聚类的问题，提出一些处理思路。

算法使用 Python 实现:

```python
class Kmeans:
    '''K-means 算法.'''

    def __init__(self, clusters, iter_times, seed):
        self.n_clusters = clusters
        self.max_iter = iter_times
        self.random_state = seed

    # 更新聚类中心
    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    # 初始化聚类中心
    def init_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    # 计算距离
    def get_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    # 寻找最近的聚类
    def assign_cluster(self, distance):
        return np.argmin(distance, axis=1)

    # 预测新样本点属于那个聚类
    def predict(self, X):
        distance = self.get_distance(X, old_centroids)
        return self.assign_cluster(distance)

    # 计算聚类的误差函数，也就是所有点和聚类中心距离平方之和
    def get_error(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    # main 函数，执行入口
    def fit(self, X):
        self.centroids = self.init_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            # 计算距离
            distance = self.get_distance(X, old_centroids)
            # 根据聚类到点的距离，得出每个元素对应最近的聚类中心
            self.labels = self.assign_cluster(distance)
            # 更新聚类中心，对每个聚类计算质心作为新的聚类中心
            self.centroids = self.compute_centroids(X, self.labels)
            # 如果新的中心和旧的相同，说明 k-means 收敛，退出循环
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.get_error(X, self.labels, self.centroids)
```

在该算法上，我们使用高斯二维分布，生成十组数据，每组数据都是由一个样本平均值和样本偏方差生成的，生成的数据使用 python 的 `matplot` 工具制图，如下：

![Image](https://pic4.zhimg.com/80/v2-98b82f6ee0af91e0569406404555bdb3.png)

数据的量级为 1000，保持数据量级不变，运行五次，统计迭代的次数和总误差.

|     | 一 | 二 | 三 | 四 | 五 |
|  ----  | ----  | --- | --- | --- | --- |
| 迭代次数  | 14 | 16 | 13 | 17 | 14 |
| 误差  | 956 | 988 | 994 | 967 | 148 |

### K-means++

在原有基础上，加上 K-means++ 算法的距离平方概率初始化，可能会减少迭代次数，优化运算的误差：

```python
np.random.RandomState(self.random_state)
# 初始化聚类中心
centroids = np.zeros((self.n_clusters, X.shape[1]));
centroids[0] = X[np.random.randint(X.shape[0])];
# 遍历每个聚类中心
for i in range(self.n_clusters-1):
    i = i+1
    # distance: [point, centroid] => distance
    distance = np.zeros((X.shape[0], i))
    # iterate through every avaliable centroids, compute distance
    for k in range(i):
        row_norm = norm(X - centroids[k, :], axis=1)
        # distance contains every point to avalable centroid
        distance[:, k] = np.square(row_norm)
    # closest_cluster: [point, closest_cluster_index];
    closest_cluster = np.argmin(distance, axis=1);
    # min_distance: [point, closest_distance]
    min_distance = np.amin(distance, axis=1);
    random_num = np.random.randint(np.sum(np.square(min_distance)));
    # iterate through every point, try to figure out which hit
    for k in range(X.shape[0]):
        random_num = random_num - np.square(min_distance[k])
        if (random_num < 0):
            # the next chosen centroid is X[k]
            centroids[i] = X[k];
            break;
```

可以看到，通过“轮盘法”，将各样本点到最近聚类中心的距离的平方作为一段区间，把长度拼接起来，在整个区间里取随机数，然后看落在哪一段，落进去的那段，就是被选中的点，因此实现了 k-means++ 所说的距离平方为概率权重的初始化。

对两种不同的算法进行统计对比，动态的生成十组二维高斯分布的样本点，样本点的数量为 3000，每组为 300 个点，然后执行 300 次，统计 *kmeans* 和 *kmeans++* 算法在误差和迭代次数上的差别。

统计代码分为两部分，第一部分是生成测试数据：

```python
def generate_samples_data(size=1000):
    means = np.array([[0, 0],
                  [3, 3],
                  [-2, -2],
                  [0, 4],
                  [3, 0],
                  [-2, 2],
                  [1, 2],
                  [2, -2],
                  [-4, 0],
                  [-4, 4]])
    samples = np.empty((0, 2))
    for i in range(10):
        X = np.random.multivariate_normal(mean=means[i],
                                          cov=[[0.5, 0], [0, 0.5]],
                                          size=size//10)                                          
        samples = np.concatenate((samples, X), 0)
    return samples
```

可以看到使用了 *numpy* 的 *mutivariate_normal* 函数用来生成二维的随机数，协方差置为 0，表示坐标之间不相关，方差都设为 0.5。

然后在原有的 *kmeans* 代码基础上，我们增加了开关参数，用来选择使用 *kmeans* 还是 *kmeans++*：

```python
if (self.kmeans_ver=='kmeans++'):
    ...# 执行 kmeans++ 的初始化
elif (self.kmeans=='kmeans'):
    ...# 执行 kmeans 的初始化
```

然后下面是测试函数：

```python
def evaluate(times=100, size=1000):

    kmeans = Kmeans(10, 100, 123, kmeans_ver='kmeans');
    kmeans_plus = Kmeans(10, 100, 123, kmeans_ver='kmeans++');
    # result_array: [iterateTimes, error];
    kmeans_result = np.zeros((times, 2));
    kmeans_plus_result = np.zeros((times, 2));
    for i in range(times):
        samples = generate_samples_data(size)
        kmeans.fit(samples)
        kmeans_plus.fit(samples)
        kmeans_result[i] = np.array([kmeans.iterate_times, kmeans.error])
        kmeans_plus_result[i] = np.array([kmeans_plus.iterate_times, kmeans_plus.error])
    np.save("kmeans", kmeans_result)
    np.save("kmeans++", kmeans_plus_result)
```

然后取平均值，生成如下的结果:

|  算法   | 平均迭代次数 | 平均误差 |
|  ----  | ----  | --- |
| *Kmeans++*  | 17.69 | 2753，68 |
| *Kmeans*  | 21.89 | 2788.03 |

可以看到，*Kmeans++* 的确显著降低了平均迭代次数，而且也一定程度降低了平均误差。

### 带离群点问题的解决：K-means--

当样本点中出现了离群点，或者噪音点后，会影响分类算法的正常工作，比如：

![Image](https://pic4.zhimg.com/80/v2-5e926c523f0de7dbbdd2b0d85a0db8f7.png)

左图的着色是按照正态分布样本生成进行分组，为本来的分组，而用 “+” 标注的点为人工生成的离群点，一共有四个，使用 *kmeans++* 算法会发现它们的存在使得分类出现了偏移，算法误把左上角的三个点分成一类，而把右下角的两类分成一类，为了解决这个问题，我们引入 *kmeans--* 算法，算法的细节在上文中已经有介绍。

K-means-- 相较于 K-means 算法，只是在聚类中心点更新的时候，将距离最近聚类中心最远的前几个忽略掉，所以经过调整后，算法为：

```python
def compute_centroids(self, X, labels, closest_distance=None):
    filter_x = X[np.argsort(closest_distance)][:-l]
    labels = labels[np.argsort(closest_distance)][:-l]
    centroids = np.zeros((self.n_clusters, X.shape[1]))
    for k in range(self.n_clusters):
        centroids[k, :] = np.mean(filter_x[labels == k, :], axis=0)
    return centroids
```

为了把前 l 个排除掉，先用 `argsort` 方法得到排序后的角标，再用切片去把最后的 l 个提取出去，得到了 `filter_x`, 再对 `label` 做同样的重拍，这样就和 `filter_x` 获得一样的排列，后面的算法与 *kmeans* 相同。

在 *kmeans--* 算法中，需要引入一个参数 l，只管意义上它应该大于等于离群点的个数，这个可以很好的降低离群点对分类的影响，使用新的算法，运行后效果如图：

![Image](https://pic4.zhimg.com/80/v2-4c992fee96c13c514b46c849b92c8e8a.png)

可以看到离群点并没有影响正常的分类，分类效果和没有离群点几乎相似。

### K-means++--

结合 *K-means++* 的快速收敛和 *K-means--* 的离群点检测，可以得到 *K-means++--* 算法，同时具有收敛快，分类效果好，抗噪声的特点。

## 参考

>[k-means--: A unified approach to clustering and outlier detection](http://users.ics.aalto.fi/gionis/kmmm.pdf)

>[k-means++: The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)


