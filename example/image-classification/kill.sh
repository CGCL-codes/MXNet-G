kill -9 $(ps aux | grep "train" | awk '{print $2}')
