#pengjing 2018.6.7

import getpass
import math

class dataDistribution(object):
    def __init__(self, Ip, speed, perc_b, perc_e, nodeNum):
        self.Ip = Ip
        self.speed = speed
        self.perc_b = perc_b
        self.perc_e = perc_e
        self.nodeNum = nodeNum

def main():
    user = getpass.getuser()
    fhost = open("/home/"+user+"/MXNet-G/example/image-classification/hosts")
    fspeed = open("/home/"+user+"/MXNet-G/example/image-classification/clusting/pretrain_speed.txt")
    fgroup = open("/home/"+user+"/MXNet-G/example/image-classification/groups")


    host = []
    speed = []
    group = []
 
    while 1:
        line = fhost.readline() 
        if not line:
            break
        if not line.isspace():
            host.append(line.strip())
    #print host

    while 1:
        line = fspeed.readline()
        if not line:
            break
        if not line.isspace():
            speed.append(float(line.strip()))
    #print speed

    while 1:
        line = fgroup.readline()
        if not line:
            break
        if not line.isspace():
            group.append(line.strip())
    #print group

    fhost.close()
    fspeed.close()
    fgroup.close()
    
    t = 0
    checkSum = 0
    dist = []
    for i in range(len(group)):
        check = 1000
        groupi = group[i].split()
        for j in range(len(groupi)):
            index = host.index(groupi[j].strip())
            ip = host[index]
            sp = speed[index]
            if check > sp:
                check = sp
            _d = dataDistribution(ip,sp,0,0,len(groupi))
            dist.append(_d)
            t = t + 1
        for j in range(t-len(groupi),t):
            dist[j].speed = check
        checkSum = checkSum + check * len(groupi)


    for i in range(len(dist)):
        if i == len(dist) - dist[len(dist) - 1].nodeNum:
            if i == 0:
                #inter = round(1.00 / dist[len(dist) - 1].nodeNum, 2)
                inter = math.floor(100 / dist[len(dist) -1].nodeNum) / 100
                for j in range(dist[i].nodeNum):
                    if j == 0:
                        dist[i + j].perc_b = 0
                    else:
                        dist[i + j].perc_b = dist[i + j - 1].perc_e
                    dist[i + j].perc_e = dist[i + j].perc_b + inter
            else:
                #inter = round((1 - dist[i - 1].perc_e) / dist[len(dist) - 1].nodeNum, 2)
                inter = math.floor(100 * (1-dist[i-1].perc_e) / dist[len(dist)-1].nodeNum) /100

                for j in range(dist[i].nodeNum):
                    dist[i + j].perc_b = dist[i + j - 1].perc_e
                    dist[i + j].perc_e = dist[i + j].perc_b + inter
            break
        elif i > 0 and i < (len(dist) - dist[len(dist) - 1].nodeNum):
            dist[i].perc_b = dist[i - 1].perc_e
            dist[i].perc_e = dist[i].perc_b + round((dist[i].speed) / checkSum, 2)
        elif i==0:
            dist[i].perc_b = 0
            dist[i].perc_e = round((dist[i].speed) / checkSum, 2)  

    
    import operator
    cmpfun = operator.attrgetter('Ip')
    dist.sort(key=cmpfun,reverse=True)

    '''
    for i in range(len(dist)):
        print dist[i].Ip
        print dist[i].perc_b
        print dist[i].perc_e
        print '\n' 
    '''

    fdata = open("/home/"+user+"/MXNet-G/example/image-classification/data_sharding",'w')
    for i in range(len(dist)):
        begin = float(dist[i].perc_b)
        end = float(dist[i].perc_e)
        fdata.write(str(begin)+' '+str(end)+'\n')


if __name__=="__main__":
	main()
