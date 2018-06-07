#2017.05  zhujian
#filter out the speed of all nodes from the log file,
#compute the average speed,then write it in .txt file
import numpy as np
import re
import getpass

def filter_out_Node(lineInfo):
	w1="Node"
	w2=" Epoch"
	start = lineInfo.find(w1)
	start += len(w1)
	end = lineInfo.find(w2)
	return lineInfo[start:end].strip('[]')

def filter_out_speed(lineInfo):
	w1="Speed: "
	w2="samples"
	start = lineInfo.find(w1)
	if start >= 0:
		start += len(w1)
		end = lineInfo.find(w2 ,start)
		if end >= 0:
			#print lineInfo[start:end].strip()
			return filter_out_Node(lineInfo),lineInfo[start:end].strip()
		else:
			return False,False
	else:
		return False,False

def main():
        user = getpass.getuser()
	ob = open("/home/"+user+"/MXNet-G/example/image-classification/clusting/pretrain.log")
	data=[]
	
	node="0"
	for line in ob.readlines():
		nodeNum,speed = filter_out_speed(line)
		if nodeNum!=False:
			if node.find(nodeNum) ==-1:
				node=node+str(nodeNum)
			data.append([int(nodeNum),float(speed)])
	#print len(node),len(data)
	data.sort(reverse=True)
        #print data
	n=0
        #ii = 0
	f = open("/home/"+user+"/MXNet-G/example/image-classification/clusting/pretrain_speed.txt",'w')
	for i in range(1,len(data)+1):
                #ii = i
                if i == len(data):
                        speed = np.mean(data[n:i],axis=0)
                        #print speed
                        f.write(str(speed[1])+"\n")
                        break
		if data[i][0]!= data[i-1][0]:#|data[i+1][0]==False:
			#print data[n:i]
                        speed = np.mean(data[n:i],axis=0)
			f.write(str(speed[1])+"\n")
                        # print speed
			#f.write(str(np.mean(data[n:i],axis=0))+"\n")
			n=i
        '''
        ii = ii + 1
        speed = np.mean(data[n:ii],axis=0)
        #print speed
	f.write(str(speed[1])+"\n")
        '''
	ob.close()
	f.close()

if __name__=="__main__":
	main()
