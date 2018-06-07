import paramiko

nodes = [16, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33]
ip = '10.10.10.'
hostname = None
username = 'ygy'
password = 'blazer'
cmd = 'nvidia-smi'
#cmd = 'htop'
#cmd = 'locate libmxnet.so'
#cmd = "ls /usr/local/cuda/lib64 | grep '*mxnet*'"
#cmd = "echo 'export LD_LIBRARY_PATH=/home/yegeyan/cuda-mxnet/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc"
paramiko.util.log_to_file('ssh_login.log')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
for i in range(len(nodes)):
    hostname = ip + str(nodes[i])
    ssh.connect(hostname=hostname, username=username, password=password)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print('nodes' + str(nodes[i]) + '  #######################################')
    print(stdout.read())

ssh.close()
