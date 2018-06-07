import os
import threading
import getpass
from sets import Set
import socket
import struct
import fcntl

def scp_groups(ip, user):
    from_path = '/home/' + user + '/MXNet-G'
    to_path = user + '@' + ip + ':' + '/home/' + user
    cmd = 'scp -r' + ' ' + from_path + ' ' + to_path
    os.system(cmd)

if __name__ == '__main__':
    nodes = []
    with open('./hosts', 'r') as file:
        nodes = [line.strip() for line in file]
    nodes = list(Set(nodes))

    interface = 'ib0'
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inet = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', interface[:15]))
    local_ip = socket.inet_ntoa(inet[20:24])
    nodes.remove(local_ip)

    user = getpass.getuser()
    nodes_num = len(nodes)
    threads = [None] * nodes_num
    for i in range(nodes_num):
        threads[i] = threading.Thread(target=scp_groups, args=(nodes[i], user))
        threads[i].start()

    for i in range(nodes_num):
        threads[i].join()
        print("Scp to " + nodes[i] + " success.")

    print('Scp Done!')
