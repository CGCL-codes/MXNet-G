# 2017.5.23 zhujian
# Hierarchical Clustering: initial points
# Kmeans: do clusting in [2,(n-1)]
#VRC: choose the best k and cluster based on the CH
import numpy as np
import random
import re
import getpass

class node:
    def __init__(self, No, ip, speed):
        self.No = No
        self.ip = ip
        self.speed = speed

class bicluster:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left = left
        self.right = right
        self.id = id
        self.vec = vec
        self.distance = distance

def yezi(clust):
    if clust.left == None and clust.right ==None:
        return [clust.id]
    return yezi(clust.left) + yezi(clust.right)

def hcluster(dataSet,n):
    biclusters = [bicluster(vec = dataSet[i],id = i) for i in range(len(dataSet))]
    distances = {}
    flag = None
    currentclusted = -1
    #clusters = []
    while(len(biclusters) > n):
        min_val = 1000000;
        biclusters_len = len(biclusters)
        #calculate the distance of any two points
        #for i in range(biclusters_len - 1):
        for i in range(biclusters_len-1):
            for j in range(i + 1 , biclusters_len):
                if distances.get((biclusters[i].id,biclusters[j].id)) == None:
                    distances[(biclusters[i].id, biclusters[j].id)] = abs(biclusters[i].vec - biclusters[j].vec)
                #find min distance
                d = distances[(biclusters[i].id,biclusters[j].id)]
                if d < min_val :
                    min_val = d
                    flag = (i,j)
        bic1,bic2 = flag
        newvec = (biclusters[bic1].vec + biclusters[j].vec)/2
        newbic = bicluster(newvec,left=biclusters[bic1], right=biclusters[bic2], distance=min_val, id=currentclusted)
        currentclusted -= 1
        del biclusters[bic2]
        del biclusters[bic1]
        biclusters.append(newbic)
        clusters = [yezi(biclusters[i]) for i in range(len(biclusters))]
        #print clusters
    #return biclusters,clusters
    return clusters

def initpoint(dataSet,k):
    initClusters = hcluster(dataSet,k)
    #print "******************",initClusters
    cluster_center = np.zeros(k)
    #print cluster_center
    for i in range(k):
        #clusterPoint = dataSet[np.nonzero(initClusters[i])]
        clusterPoint = dataSet[initClusters[i]]
        #print clusterPoint
        cluster_center[i] = np.mean(clusterPoint)
    #print "=================",cluster_center
    return cluster_center

#data:numpy.array dataset
#k:the number of cluster
def k_means(dataSet,k):

    #random generate clusteer_center
    sample_num=dataSet.shape[0]
    cluster_cen=initpoint(dataSet,k)
    
    is_change=1
    cat=np.zeros(sample_num)
 
    while is_change:
        is_change=0

        for i in range(sample_num):
            min_distance=100000
            min_index=0

            for j in range(k):
                distance=abs(dataSet[i]-cluster_cen[j])

                #print "distance----------",distance
                if distance<min_distance:
                    min_distance=distance
                    min_index=j

            if cat[i]!=min_index:
                is_change=1
                cat[i]=min_index
            #print "cat:",cat

        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(cat[:] == j)]
            #print "nonzero:",np.nonzero(cat[:]==j)
            cluster_cen[j]=np.mean(pointsInCluster)
    #print "cluster_cen:",cluster_cen
    return cat,cluster_cen

def VRC(dataSet):
    m=np.mean(dataSet)#the centroid of the entire data set
    sample_num=dataSet.shape[0]
    
    vrc = []

    for k in range(2,sample_num):
        cat, cluster_cen = k_means(dataSet,k)
        cat2 = cat.tolist()
        ssb = 0
        ssw = 0
        for i in range(k):
            nk = cat2.count(i)#nk is the number of points in cluster k
            ssb = ssb + nk*(cluster_cen[i]-m)*(cluster_cen[i]-m)

            pointsInCluster = dataSet[np.nonzero(cat[:] == i)]
            ssw = ssw + nk*(np.var(pointsInCluster))
        vrc.append((ssb*(sample_num-k))/(ssw*(k-1)))
    #print "------------vrc-------------:\n",vrc

    return (vrc.index(max(vrc))+2)

def main():
    user = getpass.getuser()
    '''
    hosts = open("/home/"+user+"/MXNet-G/example/image-classification/clusting/NumIP.txt")
    comSpeed=open("/home/"+user+"/MXNet-G/example/image-classification/clusting/node_speed.txt")

    node_speed_IP = []
    node_speed_s = []
    while 1:
        line = comSpeed.readline()
        if not line:
            break
        lline = line.split(',')
        node_speed_IP.append(lline[0].strip())
        node_speed_s.append(float(lline[1].strip()))

    host = []

    if node_speed_IP:
        while 1:
            line = hosts.readline()
            if not line :
                break
            lline = line.split(',')
            host.append(lline[1].strip())


    node_speed = []

    for i in range(len(host)):  
        index = node_speed_IP.index(host[i])
        #print index
        node_speed.append(node_speed_s[index])
        #print node_speed_s[index]

    dataSet = np.array(node_speed)
    #print dataSet
    k=VRC(dataSet)
    
    mycat,mycluster_cen=k_means(dataSet,k)
    #print mycat
    for i in range(k):
        print [index+1 for index,item in enumerate(mycat) if item == i]
    '''
    #pengjing 2018.6.6 begin
    cur_host = []
    cur_speed = []
    cur_hosts = open("/home/"+user+"/MXNet-G/example/image-classification/hosts")
    cur_speeds = open("/home/"+user+"/MXNet-G/example/image-classification/clusting/pretrain_speed.txt")

    while 1:
        cline = cur_hosts.readline()
        if not cline :
            break
        if not cline.isspace():
            cur_host.append(cline.strip())

    cur_host.sort()
    #print cur_host

    while 1:
        cline = cur_speeds.readline()
        if not cline :
            break
        if not cline.isspace():
            cur_speed.append(float(cline.strip()))
    
    # print cur_speed
     
    dataSet = np.array(cur_speed)
    k = 3
    mycat,mycluster_cen = k_means(dataSet,k)
    grouping = open("/home/"+user+"/MXNet-G/example/image-classification/groups",'w')
    for i in range(k):
        groupi = [cur_host[index] for index,item in enumerate(mycat) if item == i]
        for group_ip in groupi:
            grouping.write(str(group_ip)+' ')
        grouping.write('\n')
    grouping.close()


    #pengjing 2018.6.6 end


if __name__=="__main__":
    main()


