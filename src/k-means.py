import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


import numpy as np
from numpy.linalg import norm


class Kmeans:
    '''K-means 算法.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123, kmeans_ver='kmeans++', n_outliners=3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.kmeans_ver = kmeans_ver
        self.iterate_times = 0
        self.l = n_outliners

    # 初始化聚类中心
    def init_cents(self, X):
        if (self.kmeans_ver == 'kmeans++' or self.kmeans_ver == 'kmeans++--'):
            np.random.RandomState(self.random_state)
            # initialize centroids
            centroids = np.zeros((self.n_clusters, X.shape[1]))
            centroids[0] = X[np.random.randint(X.shape[0])]
            # iterate through every centroids
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
                closest_cluster = np.argmin(distance, axis=1)
                # min_distance: [point, closest_distance]
                min_distance = np.amin(distance, axis=1)
                random_num = np.random.randint(np.sum(np.square(min_distance)))
                # iterate through every point, try to figure out which hit
                for k in range(X.shape[0]):
                    random_num = random_num - np.square(min_distance[k])
                    if (random_num < 0):
                        # the next chosen centroid is X[k]
                        centroids[i] = X[k]
                        break
        elif (self.kmeans_ver == 'kmeans' or self.kmeans_ver == 'kmeans--'):
            np.random.RandomState(self.random_state)
            random_idx = np.random.permutation(X.shape[0])
            centroids = X[random_idx[:self.n_clusters]]

        return centroids

    # 更新聚类中心
    def update_cents(self, X, labels, closest_distance=None):
        if self.kmeans_ver == 'kmeans++' or self.kmeans_ver == 'kmeans':
            cents = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                cents[k, :] = np.mean(X[labels == k, :], axis=0)
            return cents
        elif self.kmeans_ver == 'kmeans++--' or self.kmeans_ver == 'kmeans--':
            filter_x = X[np.argsort(closest_distance)][:-self.l]
            labels = labels[np.argsort(closest_distance)][:-self.l]
            cents = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                cents[k, :] = np.mean(filter_x[labels == k, :], axis=0)
            return cents

    # 计算距离
    def get_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def find_closest_distance(self, distance):
        return np.amin(distance, axis=1)

    # 计算聚类的误差函数，也就是所有点和聚类中心距离平方之和
    def get_error(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    # main 函数，执行入口
    def fit(self, X):
        self.centroids = self.init_cents(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.get_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.closest_distance = self.find_closest_distance(distance)
            self.centroids = self.update_cents(
                X, self.labels, self.closest_distance)
            if np.all(old_centroids == self.centroids):
                self.iterate_times = i
                break
        self.error = self.get_error(X, self.labels, self.centroids)

    def predict(self, X):
        distance = self.get_distance(X, old_centroids)
        return self.find_closest_cluster(distance)


def plot_random_data(ifOutliner=False):
    '''画出随机生成的数据'''
    means = np.array([[0, 0],
                      [3, 4],
                      [-2, -2],
                      [0, 4],
                      [3, 0],
                      [-2, 2],
                      [1, 2],
                      [2, -2],
                      [-4, 0],
                      [-4, 4]])
    outliners = np.array([[-7, 9], [-8, 5], [6, 6], [-6, -7]])
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 8)
    plt.subplot(121)
    samples = np.empty((0, 2))
    for i in range(10):
        X = np.random.multivariate_normal(mean=means[i],
                                          cov=[[0.5, 0], [0, 0.5]],
                                          size=100)
        samples = np.concatenate((samples, X), 0)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[i])
    if (ifOutliner):
        samples = np.concatenate((samples, outliners))
        plt.scatter(outliners[:, 0], outliners[:, 1],
                    s=40, color="white", marker='+')
    plt.xlim([-9, 8])
    ax.set_aspect('equal')
    ax.grid('false')
    # print(samples)
    return samples
    # plt.show()


def visualize(onlinerOn=False):
    colors = [
        'orange', 'red', 'cyan', 'green', 'blue', 'magenta',
        'lightgreen', 'yellow', 'lightyellow', 'lightblue', 'white'
    ]

    kmeans = Kmeans(10, 100, 123, kmeans_ver='kmeans', n_outliners=4)
    samples = plot_random_data(onlinerOn)
    kmeans.fit(samples)
    # centroids = kmeans.initializ_centroids(samples)
    # plt.scatter(samples[:, 0], samples[:, 1], s=20, color="white", marker='o')

    plt.subplot(122)
    for i in range(10):
        plt.scatter(
            kmeans.centroids[i, 0], kmeans.centroids[i, 1], s=100, color=colors[i], marker='^')
        plt.scatter(samples[kmeans.labels == i, 0],
                    samples[kmeans.labels == i, 1], s=20, color=colors[i], marker='o')

    print(f"error: {kmeans.error}")
    plt.show()


def generate_samples_data(size=1000):
    means = np.array([[0, 0], [3, 3], [-2, -1], [0, 5], [3, 0], [-2, 2], [1, 2],
            [3, -3], [-4, 0], [-4, 4]])
    samples = np.empty((0, 2))
    for i in range(10):
        X = np.random.multivariate_normal(mean=means[i],
                                          cov=[[0.5, 0], [0, 0.5]],
                                          size=size//10)
        samples = np.concatenate((samples, X), 0)
    return samples


def evaluate(times=100, size=1000):

    kmeans = Kmeans(10, 100, 123, kmeans_ver='kmeans')
    kmeans_plus = Kmeans(10, 100, 123, kmeans_ver='kmeans++')
    # result_array: [iterateTimes, error];
    kmeans_result = np.zeros((times, 2))
    kmeans_plus_result = np.zeros((times, 2))
    for i in range(times):
        samples = generate_samples_data(size)
        kmeans.fit(samples)
        kmeans_plus.fit(samples)
        kmeans_result[i] = np.array([kmeans.iterate_times, kmeans.error])
        kmeans_plus_result[i] = np.array(
            [kmeans_plus.iterate_times, kmeans_plus.error])
    np.save("kmeans", kmeans_result)
    np.save("kmeans++", kmeans_plus_result)


# evaluate(300, size=3000)
visualize(False)
