import os
from flask import Flask, request, render_template, send_from_directory
from sklearn import cluster
from pyspark.mllib.clustering import KMeans
import numpy as np
from pyspark import SparkContext
from shutil import copyfile
import time
import matplotlib.pyplot as plt

__author__ = 'aungkon'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    k = int(request.form['projectFilepath'])

    target = os.path.join(APP_ROOT, 'data/')
    # target = os.path.join(APP_ROOT, 'static/')
    # print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    # print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        copyfile("data/"+filename,filename)
    normal_secs,centers_normal = run_normal(k,filename)
    centers_normalx = []
    centers_normaly = []
    for element in centers_normal:
        centers_normalx.append(element[0])
        centers_normaly.append(element[1])
    os.system("/home/aungkon/hadoop-2.9.0/sbin/stop-dfs.sh")
    os.system("rm -Rf /tmp/hadoop-aungkon/*")
    os.system("yes | /home/aungkon/hadoop-2.9.0/bin/hdfs namenode -format")
    os.system("yes | /home/aungkon/hadoop-2.9.0/bin/hdfs datanode -format")
    os.system("/home/aungkon/hadoop-2.9.0/sbin/start-dfs.sh")
    os.system("/home/aungkon/hadoop-2.9.0/bin/hdfs dfs -mkdir /user")
    os.system("/home/aungkon/hadoop-2.9.0/bin/hdfs dfs -mkdir /user/aungkon")
    os.system("/home/aungkon/hadoop-2.9.0/bin/hdfs dfs -put " + os.path.join(APP_ROOT, 'data')+" input")
    start = time.time()
    os.system("/home/aungkon/hadoop-2.9.0/bin/hadoop jar kmeans.jar kmeans input output "+str(k)+" "+ os.path.join(APP_ROOT, 'data'))
    end = time.time()
    os.system("rm -Rf output")
    os.system("/home/aungkon/hadoop-2.9.0/bin/hdfs dfs -get output output")
    os.system("/home/aungkon/hadoop-2.9.0/sbin/stop-dfs.sh")
    centers_spark, spark_secs = run_in_spark(k)
    centers_sparkx = []
    centers_sparky = []
    for g in centers_spark:
        centers_sparkx.append(g[0])
        centers_sparky.append(g[1])
    hadoop_secs = end - start
    data = np.genfromtxt(filename,delimiter=",")
    pointx = []
    pointy = []
    for element in data:
        pointx.append(element[0])
        pointy.append(element[1])
    center_hadoop = []
    center_hadoopx = []
    center_hadoopy  = []
    with open("output/part-r-00000",'rt') as f:
        for l in f:
            l= l.split("\n")[0]
            x = l.split("\t")[1].split(",")
            center_hadoop.append([int(x[0]),int(x[1])])
            center_hadoopx.append(int(x[0]))
            center_hadoopy.append(int(x[1]))
    # Turn interactive plotting off
    plt.ioff()
    # Create a new figure, plot into it, then close it so it never gets displayed
    fig = plt.figure()
    plt.scatter(pointx,pointy,label = "Original Datapoints")
    plt.scatter(center_hadoopx,center_hadoopy,label = "Cluster Centers From Hadoop Job")
    plt.scatter(centers_sparkx,centers_sparky,label = "Cluster Centers From Spark Job")
    plt.scatter(centers_normalx,centers_normaly,label = "Cluster Centers From Standard K means algorithm")
    plt.legend(loc='best')
    plt.savefig('templates/test0.png')
    plt.close(fig)
    time.sleep(10)
    return render_template("complete.html",normal_secs = normal_secs,spark_secs = spark_secs,hadoop_secs = hadoop_secs)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("data", filename)

def run_in_spark(k:int):
    sc = SparkContext("local", "K-means")
    start = time.time()
    lines = sc.textFile("data.txt")
    data = lines.map(lambda line: np.array([float(x) for x in line.split(',')]))
    model = KMeans.train(data, k)
    end = time.time()
    centers = []
    for x in model.clusterCenters:
        centers.append(x)
    sc.stop()
    print(centers)
    spark_secs = end-start
    return centers, spark_secs

def run_normal(k,filename):
    start = time.time()
    data = np.genfromtxt(filename,delimiter=",")
    # centroids = data[:k]
    k_means = cluster.KMeans(n_clusters=k)
    k_means.fit(data)
    end = time.time()
    return end-start,k_means.cluster_centers_


if __name__ == "__main__":
    app.run(port=4555, debug=True)
