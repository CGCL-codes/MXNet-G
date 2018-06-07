import socket
import struct
import fcntl

def getIp(ethname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0X8915, struct.pack('256s', ethname[:15]))[20:24])

def getWorkerNumInGroup(ip):
    fr = open("groups")
    for line in fr.readlines():
        lineArr = line.strip().split()
        if ip in lineArr:
            return len(lineArr)
    print("Not found this ip:{}".format(ip))
    return 0

if __name__ == '__main__':
    workerNumInGroup = getWorkerNumInGroup(getIp('ib0'))
    print(workerNumInGroup)
