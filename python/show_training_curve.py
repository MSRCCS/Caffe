from numpy  import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import rand
import os
import math

def PlotTrainingLog(logfile):
    log = open(logfile, 'r')
    iterList = []
    lossList = []
    iterTestList = []
    testScoreList0 = []
    testScoreList1 = []
    itrLrList = []
    lrList = []
    iter = 0
    try:
        for line in log:
            if 'Iteration ' in line and 'loss' in line:
                part = line.split(']')[1].strip()
                iteration = part.split(',')[0].strip()
                
                loss_str = part.split(',')[1].strip()

                iter = int(iteration.split(' ')[1])
                loss = float(loss_str.split('=')[1])
                iterList.append(iter)
                lossList.append(loss)
            elif 'Test net output #0' in line:
                part = line.split(']')[1].strip()
                testScore = part.split('accuracy = ')[1].strip()
                iterTestList.append(iter)
                testScoreList0.append(float(testScore) * 10)
            elif 'Test net output #1' in line:
                part = line.split(']')[1].strip()
                testScore = part.split('loss = ')[1].split('(')[0].strip()
                testScoreList1.append(float(testScore))
            elif ', lr = ' in line:
                part = line.split(']')[1].strip() # Iteration 339960, lr = 1e-005
                iteration = part.split(',')[0].strip()
                loss_str = part.split(',')[1].strip()
                iter = int(iteration.split(' ')[1])
                lr = float(loss_str.split('=')[1])
                itrLrList.append(iter)
                lrList.append(math.log10(float(lr)))
    except:
        print ' '

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
    
    

