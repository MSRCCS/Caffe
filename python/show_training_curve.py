from numpy  import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import rand
import os
import math
from datetime import datetime

def PlotTrainingLog(logfile):
    log = open(logfile, 'r')
    iterList = []
    lossList = []
    iterStartTimeList = []
    iterTimeCostList = []
    iterTestList = []
    testScoreList0 = []
    testScoreList1 = []
    itrLrList = []
    lrList = []
    iter = 0
    start_time = datetime.now()
    start_time_year = start_time.year
    try:
        for line in log:
            if 'Log file created' in line:
                #Log file created at: 2016/02/09 18:21:04
                start_time_str = line.split('at: ')[1].strip()
                start_time = datetime.strptime(start_time_str, "%Y/%m/%d %H:%M:%S")
                start_time_year = start_time.year
                iterStartTimeList.append(start_time)
            elif 'Iteration ' in line and 'loss' in line:
                #I0128 14:07:16.713331 21300 solver.cpp:237] Iteration 0, loss = 8.86471
                part = line.split(']')[1].strip()
                iteration = part.split(',')[0].strip()
                loss_str = part.split(',')[1].strip()
                iter = int(iteration.split(' ')[1])
                loss = float(loss_str.split('=')[1])
                iterList.append(iter)
                lossList.append(loss)
                iter_start_time_str = line.split('.')[0]
                #standard caffe training log line doesn't have year number, so we need to guess it according to the first Log line.
                iter_start_time = datetime.strptime( str(start_time_year) + iter_start_time_str, "%YI%m%d %H:%M:%S")
                if iter_start_time < iterStartTimeList[-1]: #this can happen if training across new year's eve
                    start_time_year += 1
                    iter_start_time = datetime.strptime( str(start_time_year) + iter_start_time_str, "%YI%m%d %H:%M:%S")
                iterStartTimeList.append(iter_start_time)
                iterTimeCostList.append(int((iterStartTimeList[-1] - iterStartTimeList[-2]).total_seconds()))    
            elif 'Test net output #0' in line:
                #I0128 14:09:49.408828 21300 solver.cpp:409]     Test net output #0: accuracy = 1
                part = line.split(']')[1].strip()
                testScore = part.split('accuracy = ')[1].strip()
                iterTestList.append(iter)
                testScoreList0.append(float(testScore) * 10)
            elif 'Test net output #1' in line:
                #I0128 14:09:49.409827 21300 solver.cpp:409]     Test net output #1: loss = 0 (* 1 = 0 loss)
                part = line.split(']')[1].strip()
                testScore = part.split('loss = ')[1].split('(')[0].strip()
                testScoreList1.append(float(testScore))
            elif ', lr = ' in line:
                #I0128 14:09:49.549926 21300 sgd_solver.cpp:106] Iteration 500, lr = 0.002
                part = line.split(']')[1].strip() # Iteration 339960, lr = 1e-005
                iteration = part.split(',')[0].strip()
                loss_str = part.split(',')[1].strip()
                iter = int(iteration.split(' ')[1])
                lr = float(loss_str.split('=')[1])
                itrLrList.append(iter)
                lrList.append(math.log10(float(lr)))            
    except:
        print ' '

    plt.plot(iterList, iterTimeCostList, 'y-', label='time cost(sec)')
    plt.plot(iterList, lossList, 'r-', label='train loss')    
    plt.plot(iterTestList, testScoreList1, 'b*-', label='test loss')
    plt.plot(iterTestList, testScoreList0, 'g*-', label='test accuracy * 10')
    plt.plot(itrLrList, lrList, 'k-', label='learning rate')
    return plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

def main(argv):
    import sys
    if len(argv) < 2:
        print 'Usage: %s log_file_name []out_file_name]' %os.path.basename(sys.argv[0])
    log_filename = argv[1]
    legend = PlotTrainingLog(log_filename)

    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.grid(True)
    if(len(argv)>=3):
        plt.savefig(argv[2], bbox_extra_artists=(legend,), bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    import sys
    main(sys.argv)
    
    

