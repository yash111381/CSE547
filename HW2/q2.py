import re
import sys
from pyspark import SparkConf, SparkContext
from heapq import nlargest
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from operator import add

conf = SparkConf()
sc = SparkContext(conf = conf)

def func(line):
    parts = line.split(" ")
    return list(map(float,parts))

lines = sc.textFile(sys.argv[1])
setOfDocuments = lines.map(func)

initialRandomCentroidsFile = sc.textFile(sys.argv[2])
initialRandomCentroids = initialRandomCentroidsFile.map(func)
initialRandomCentroids_actual = initialRandomCentroids.collect()
initialFarCentroidsFile = sc.textFile(sys.argv[3])
initialFarCentroids = initialFarCentroidsFile.map(func)
initialFarCentroids_actual = initialFarCentroids.collect()

error_vals = []

def computeDistanceFromRandomCentroids(point, initialRandomCentroids_actual):
  dict1 = {}
  for centroid in initialRandomCentroids_actual:
    dict1[tuple(centroid)] = np.linalg.norm(np.array(point)-np.array(centroid),2)
  sortedList = sorted(dict1.items(), key = lambda x:x[1])
  return (sortedList[0][0], point)

def normDiff(points):
  return np.linalg.norm(np.array(points[0])-np.array(points[1]),2)

def calculateCost(pointClusterRdd):
  normsRdd = pointClusterRdd.map(normDiff)
  return normsRdd.reduce(add)

def computeNewCentroid(rddVals):
  sumOfPoints = []
  for val in rddVals:
    for i,pointInDim in enumerate(val):
      if len(sumOfPoints)<i+1:
        sumOfPoints.append(pointInDim)
      else:
        sumOfPoints[i] = sumOfPoints[i]+pointInDim
  return list((1/len(rddVals))*np.array(sumOfPoints))
  
current_centroids = initialRandomCentroids_actual
for i in range(20):
  pointClusterRdd = setOfDocuments.map(lambda x:computeDistanceFromRandomCentroids(x, current_centroids))
  error_vals.append(calculateCost(pointClusterRdd))
  oldNewCentroids = pointClusterRdd.groupByKey().mapValues(computeNewCentroid)
  newCentroids = oldNewCentroids.map(lambda x:x[1])
  current_centroids = newCentroids.collect()
  
plt.title("Cost vs. number of iterations for random initialization L2 norm")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
_ = plt.plot(list(range(1,21)), error_vals)
plt.savefig("L2Random.jpg")

error_vals_far = []

current_centroids = initialFarCentroids_actual
for i in range(20):
  pointClusterRdd = setOfDocuments.map(lambda x:computeDistanceFromRandomCentroids(x, current_centroids))
  error_vals_far.append(calculateCost(pointClusterRdd))
  oldNewCentroids = pointClusterRdd.groupByKey().mapValues(computeNewCentroid)
  newCentroids = oldNewCentroids.map(lambda x:x[1])
  current_centroids = newCentroids.collect()
  
plt.title("Cost vs. number of iterations for far initialization L2 norm")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
_ = plt.plot(list(range(1,21)), error_vals_far)
plt.savefig("L2far.jpg")

percentChangeRandom = abs(error_vals[9]-error_vals[0])/error_vals[0]
percentChangeFar = abs(error_vals_far[9]-error_vals_far[0])/error_vals_far[0]

print("Percentage change for c1.txt i.e. Random cluster centroid initialization",percentChangeRandom)
print("Percentage change for c2.txt i.e. Far cluster centroid initialization",percentChangeFar)

#As we can see, the percentage change for the initialization of centroids using c2.text is far more than that using the centroids from c1.txt.
#The initialization using c2.txt is better because using random initialization might initialize clusters near to each other, i.e. points in different clusters might not be that different compared to when initialized using points as far as possible. (Note that when initializing cluster centroids using points as far as possible, we should take care that there are not outliers in the data as it can harm our clustering)

error_vals_man = []

def computeDistanceFromRandomCentroids_man(point, initialRandomCentroids_actual):
  dict1 = {}
  for centroid in initialRandomCentroids_actual:
    dict1[tuple(centroid)] = np.linalg.norm(np.array(point)-np.array(centroid),1)
  sortedList = sorted(dict1.items(), key = lambda x:x[1])
  return (sortedList[0][0], point)

def normDiff_man(points):
  return np.linalg.norm(np.array(points[0])-np.array(points[1]),1)

def calculateCost_man(pointClusterRdd):
  normsRdd = pointClusterRdd.map(normDiff_man)
  return normsRdd.reduce(add)
  
current_centroids = initialRandomCentroids_actual
for i in range(20):
  pointClusterRdd = setOfDocuments.map(lambda x:computeDistanceFromRandomCentroids_man(x, current_centroids))
  error_vals_man.append(calculateCost_man(pointClusterRdd))
  oldNewCentroids = pointClusterRdd.groupByKey().mapValues(computeNewCentroid)
  newCentroids = oldNewCentroids.map(lambda x:x[1])
  current_centroids = newCentroids.collect()
  
plt.title("Cost vs. number of iterations for random initialization L1 norm")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
_ = plt.plot(list(range(1,21)), error_vals_man)
plt.savefig("l1random.jpg")

error_vals_man_far = []

current_centroids = initialFarCentroids_actual
for i in range(20):
  pointClusterRdd = setOfDocuments.map(lambda x:computeDistanceFromRandomCentroids_man(x, current_centroids))
  error_vals_man_far.append(calculateCost_man(pointClusterRdd))
  oldNewCentroids = pointClusterRdd.groupByKey().mapValues(computeNewCentroid)
  newCentroids = oldNewCentroids.map(lambda x:x[1])
  current_centroids = newCentroids.collect()
  
  
plt.title("Cost vs. number of iterations for far initialization L1 norm")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
_ = plt.plot(list(range(1,21)), error_vals_man_far)
plt.savefig("l1far.jpg")

percentChangeRandom_man = abs(error_vals_man[9]-error_vals_man[0])/error_vals_man[0]
percentChangeFar_man = abs(error_vals_man_far[9]-error_vals_man_far[0])/error_vals_man_far[0]

print("Percentage change for c1.txt i.e. Random cluster centroid initialization",percentChangeRandom_man)
print("Percentage change for c2.txt i.e. Far cluster centroid initialization",percentChangeFar_man)

sc.stop()
