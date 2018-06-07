#!/usr/bin/env python

import os, sys
import threading
from sets import Set

if len(sys.argv) != 2:
  print "usage: %s <hostfile>" % sys.argv[0]
  sys.exit(1)

host_file = sys.argv[1]
prog_name = "train"

# Get host IPs
with open(host_file, "r") as f:
  hosts = f.read().splitlines()
hosts = list(Set(hosts))
ssh_cmd = (
    "ssh "
    "-o StrictHostKeyChecking=no "
    "-o UserKnownHostsFile=/dev/null "
    "-o LogLevel=quiet "
    )
kill_cmd = (
    " "
    "ps aux |"
    "grep -v grep |"
    "grep 'train' |"
    "awk '{print \$2}'|"
    "xargs kill"
    )
print kill_cmd

#author yegeyan
def kill_workers(host):
    cmd = ssh_cmd + host +" \""+ kill_cmd+"\""
    print cmd
    os.system(cmd)

nodes_num = len(hosts)
threads = [None] * nodes_num
for i in range(nodes_num):
    threads[i] = threading.Thread(target=kill_workers, args=(hosts[i],))
    threads[i].start()
    
for i in range(nodes_num):
    threads[i].join()
    print "Done killing"
