import datetime
import time
from datetime import datetime


def process_tasks(taskRawFile, taskCsvFile):
    jobInfoList = ''
    for line in open(taskRawFile):
        jobInfo = line.rstrip('\n').split('|')
        id = jobInfo[0]
        try:
            submitTime = jobInfo[10]
            startTime = jobInfo[11]
            endTime = jobInfo[12]
        except:
            continue
        if '.' in id:
            continue
        if submitTime == 'Unknown' or startTime == 'Unknown': #or endTime == 'Unknown':
            continue
        gpuNum = 0
        try:
            gpuNum = int(jobInfo[6].split(':')[1])
        except Exception:
            pass
        jobInfo.append(gpuNum)

        submitTime = datetime.strptime(submitTime, '%Y-%m-%dT%H:%M:%S')
        startTime = datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S')
        if endTime!='Unknown':
            endTime = datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%S')
            runTime = int((endTime - startTime).total_seconds())
        else:
            runTime = -1

        pdTime = int((startTime - submitTime).total_seconds())
        jobInfo.append(pdTime)
        jobInfo.append(runTime)
        jobInfo = ','.join(map(str, jobInfo))
        jobInfoList += jobInfo
        jobInfoList += '\n'

    with open(taskCsvFile, 'w') as f:
        print(jobInfoList, file=f)

if __name__== "__main__":
    process_tasks('tasks.txt', 'tasks.csv')