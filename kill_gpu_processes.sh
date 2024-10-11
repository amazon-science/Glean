#!/bin/bash

# Function to kill process and its parent
kill_process_and_parent() {
    local pid=$1
    local ppid=$(ps -o ppid= -p $pid)
    
    if [ ! -z "$ppid" ]; then
        echo "Killing parent process $ppid"
        kill -9 $ppid
    fi
    
    echo "Killing process $pid"
    kill -9 $pid
}

# Find all PIDs of GCDLLMs.py processes
pids=$(ps aux | grep GCDLLMs.py | grep -v grep | awk '{print $2}')

# Kill each GCDLLMs.py process and its parent process
for pid in $pids; do
    kill_process_and_parent $pid
done

echo "All GCDLLMs.py processes and their parent processes have been killed."
