import os
import threading

def scp_miniters(ip):
    path = '/home/yegeyan/MXNet-G/example/image-classification/'
    from_path = './ssp/miniters.log'
    to_path = 'yegeyan@' + ip + ':' + path
    cmd = 'scp' + ' ' + from_path + ' ' + to_path
    os.system(cmd)

if __name__ == '__main__':
    file = open("ssp/miniters.log", "w")
    file.write("0")
    file.close()
    nodes = [16, 22, 23, 24, 25, 27, 29, 30, 31, 32, 33]
    ip = '10.10.10.'
    nodes_num = len(nodes)
    threads = [None] * nodes_num
    for i in range(nodes_num):
        threads[i] = threading.Thread(target=scp_miniters, args=(ip + str(nodes[i]),))
        threads[i].start()

    for i in range(nodes_num):
        threads[i].join()

    print("Scp ssp/miniters.log Done!")
