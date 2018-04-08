import random
import numpy as np

def init_centroids(k, X):
    centroids = []
    for _ in range(k):
        centroids.append([random.random()*max(X[:,0]),random.random()*max(X[:,1])])
    return centroids

def train(k, X, iteration):
    centroids = init_centroids(k, X)
    
    for _ in range(iteration):
        clusters = {i:[] for i in range(k)}
        
        for row in X:
            distances = np.linalg.norm(row-centroids,axis=1)
            clusters[np.argmin(distances)].append(row)
        
        for i in range(len(centroids)):
            centroids[i] = np.mean(clusters[i], axis=0)
            
    return clusters, centroids

def predict(X, centroids):
    return [np.argmin(np.linalg.norm(row-centroids, axis=1)) for row in X]

def getSumSquaredError(centroids, clusters):
    sse = 0
    for centroid, members in clusters:
        sse += np.sum((centroids[centroid]-members) ** 2)
    return sse