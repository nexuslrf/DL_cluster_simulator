#! /bin/bash
sacct -a -S 2019-09-1 -E Now -P -o JobID,JobName,Partition,ReqNodes,ReqGRES,AllocNodes,AllocGRES,User,Priority,State,Submit,Start,End -n > tasks.txt