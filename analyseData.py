# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pdb
from Data_driver import Data_driver
from os import listdir
from os.path import isfile, join
import scipy.io as sio

from klepto.archives import file_archive

import os
from my_statistics import My_statistics

def setBoxColors(bp):
    #plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][0], facecolor='blue')
    #plt.setp(bp['caps'][0], color='blue')
    #plt.setp(bp['caps'][1], color='blue')
    #plt.setp(bp['whiskers'][0], color='blue')
    #plt.setp(bp['whiskers'][1], color='blue')
    #plt.setp(bp['fliers'][0], color='blue')
    #plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='white')

    plt.setp(bp['boxes'][1], color='red')
    #plt.setp(bp['caps'][2], color='red')
    #plt.setp(bp['caps'][3], color='red')
    #plt.setp(bp['whiskers'][2], color='red')
    #plt.setp(bp['whiskers'][3], color='red')
    #plt.setp(bp['fliers'][2], color='red')
    #plt.setp(bp['fliers'][3], color='red')
    plt.setp(bp['medians'][1], color='white')

    plt.setp(bp['boxes'][2], color='green')
    #plt.setp(bp['caps'][4], color='green')
    #plt.setp(bp['caps'][5], color='green')
    #plt.setp(bp['whiskers'][4], color='green')
    #plt.setp(bp['whiskers'][5], color='green')
    plt.setp(bp['medians'][2], color='white')

    plt.setp(bp['boxes'][3], color='black')
    #plt.setp(bp['caps'][6], color='black')
    #plt.setp(bp['caps'][7], color='black')
    #plt.setp(bp['whiskers'][6], color='black')
    #plt.setp(bp['whiskers'][7], color='black')
    plt.setp(bp['medians'][3], color='white')

    plt.setp(bp['boxes'][4], color='magenta')
    #plt.setp(bp['caps'][8], color='magenta')
    #plt.setp(bp['caps'][9], color='magenta')
    #plt.setp(bp['whiskers'][8], color='magenta')
    #plt.setp(bp['whiskers'][9], color='magenta')
    plt.setp(bp['medians'][4], color='white')

    plt.setp(bp['boxes'][4], color='teal')
    #plt.setp(bp['caps'][8], color='magenta')
    #plt.setp(bp['caps'][9], color='magenta')
    #plt.setp(bp['whiskers'][8], color='magenta')
    #plt.setp(bp['whiskers'][9], color='magenta')
    plt.setp(bp['medians'][4], color='white')

nbType = 6
myStats= np.zeros(nbType,My_statistics)
nbTrial = 10
nbPercent=5
all_dist_real_inf = np.zeros([nbType,7*nbTrial,nbPercent+1])# devra etre 6
all_dist_reconstr_inf = np.zeros([nbType,7*nbTrial,nbPercent])# devra etre 5


comparsisonLS = False
allComparison = False

listLS = ['data/AE_myLittleStats_LS_2_epochs_5000',
'data/AE_myLittleStats_LS_5_epochs_5000','data/AE_myLittleStats_LS_10_epochs_5000','data/AE_myLittleStats_LS_15_epochs_5000','data/AE_myLittleStats_LS_30_epochs_5000','data/AE_myLittleStats_LS_69_epochs_5000']#,'data/myLongStats_5LS','data/myLongStatsLS7','data/myLongStatsLS15','data/myLongStatsLS20']
#labelsName =['10','20','30','40','50','60','70','80','90','100', 'Control\ngroups']
labelsName =['20','40','60','80','100', 'Control\ngroup']
for nbSS, myLS in enumerate(listLS):
    with open(myLS, 'rb') as fichier:
        mon_dep = pickle.Unpickler(fichier)
        myStats[nbSS] = mon_dep.load()
        myStats[nbSS].get_distance()
        all_dist_real_inf[nbSS,:,0:nbPercent] = np.transpose(myStats[nbSS].mean_dist_real_inf,(1,0)) 
        
        all_dist_real_inf[nbSS,:,nbPercent] = myStats[nbSS].mean_dist_real_reconstr #conrol group
        print(str(myStats[nbSS].mean_dist_real_inf.shape))
        print(str(myStats[nbSS].mean_dist_real_reconstr.shape))
        all_dist_reconstr_inf[nbSS,:,0:nbPercent] = np.transpose(myStats[nbSS].mean_dist_reconstr_inf,(1,0)) 
    

        #all_dist_real_inf[nbSS,:,0:10] = np.transpose(myStats[nbSS].mean_dist_real_inf,(1,0)) 
        #all_dist_real_inf[nbSS,:,10] = myStats[nbSS].mean_dist_real_reconstr #conrol group
        #all_dist_reconstr_inf[nbSS,:,0:10] = np.transpose(myStats[nbSS].mean_dist_reconstr_inf,(1,0)) 









####Comparison LS
if comparsisonLS:
    
    ## Real Inf 60 percent
    fig = plt.figure()
    ax = plt.axes()
    plt.hold(True)
    
    #for i in range(nbPercent+1):
        ## i-th boxplot
        #bp = plt.boxplot(np.transpose(all_dist_real_inf[:,:,i],(1,0)), positions = [ i+1 + i*nbType, i+2+i*nbType, i+3+i*nbType,i+4+i*nbType,i+5+i*nbType, i+(i+1)*nbType], widths = 0.8,patch_artist = True)
        #setBoxColors(bp)
    
    
    
    bp = plt.boxplot(np.transpose(all_dist_real_inf[:,:,2],(1,0)), positions = [ 0+1 + 0*nbType, 2, 3,4,5, 6], widths = 0.8,patch_artist = True)
    setBoxColors(bp)
    i=1
    bp = plt.boxplot(np.transpose(all_dist_real_inf[:,:,5],(1,0)), positions = [ i+1 + i*nbType, i+2+i*nbType, i+3+i*nbType,i+4+i*nbType,i+5+i*nbType, i+(i+1)*nbType], widths = 0.8,patch_artist = True)
    setBoxColors(bp)
    
    #i=nbPercent
    #bp = plt.boxplot(np.transpose(all_dist_real_inf[:,:,i],(1,0)), positions = [ (nbPercent+1)+60, (nbPercent+1)+1+10*nbType, (nbPercent+1)+2+10*nbType,(nbPercent+1)+3+10*nbType,(nbPercent+1)+4+10*nbType, (nbPercent+1)+5+10*nbType], widths = 0.8,patch_artist = True)# [ 11+60], widths = 0.8,patch_artist = True)
    #setBoxColors(bp)
    
    # set axes limits and labels
    plt.xlim(0,1)
    plt.ylim(0,0.03)
    plt.title('Error distance between the real trajectory and the infered one' , fontsize=20)
    ax.set_xticklabels([str(60)+'%','Control group'], fontsize=20)
    ax.yaxis.label.set_size(22)
    
    midlle= 3
    ax.set_xticks([midlle,midlle + (1 + nbType),midlle*2 + (2 + nbType) ])
    ax.yaxis.set_tick_params(labelsize=20)
    
    ax.set_xlabel('Observed data [%]', fontsize=20)
    ax.set_ylabel('Distance error [m]', fontsize=20)
    # draw temporary red and blue lines and use them to create a legend
    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    hG, = plt.plot([1,1],'g-')
    hK, = plt.plot([1,1],'k-')
    hM, = plt.plot([1,1],'m-')
    hC, = plt.plot([1,1],'teal')
    plt.legend((hB, hR, hG, hK, hM,hC),('2', '5', '10', '15', '30','69'), title='Latent Space dimension', fontsize=18)
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)
    hK.set_visible(False)
    hM.set_visible(False)
    hC.set_visible(False)
    plt.show()
    
    
    
    ## Reconstr Inf 60 percent
    fig = plt.figure()
    ax = plt.axes()
    plt.hold(True)
    
    
    
    bp = plt.boxplot(np.transpose(all_dist_reconstr_inf[:,:,2],(1,0)), positions = [ 0+1 + 0*nbType, 2, 3,4,5, 6], widths = 0.8,patch_artist = True)
    setBoxColors(bp)
    
    # set axes limits and labels
    plt.xlim(0,1)
    plt.ylim(0,0.03)
    plt.title('Error distance between the reconstruction of\n the real trajectory and the infered one' , fontsize=20)
    ax.set_xticklabels([str(60)+'%'], fontsize=20)
    ax.yaxis.label.set_size(22)
    
    midlle= 3
    ax.set_xticks([midlle,midlle*2 + 1 ])
    ax.yaxis.set_tick_params(labelsize=20)
    
    ax.set_xlabel('Observed data [%]', fontsize=20)
    ax.set_ylabel('Distance error [m]', fontsize=20)
    # draw temporary red and blue lines and use them to create a legend
    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    hG, = plt.plot([1,1],'g-')
    hK, = plt.plot([1,1],'k-')
    hM, = plt.plot([1,1],'m-')
    hC, = plt.plot([1,1],'teal')
    plt.legend((hB, hR, hG, hK, hM,hC),('2', '5', '10', '15', '30','69'), title='Latent Space dimension', fontsize=18)
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)
    hK.set_visible(False)
    hM.set_visible(False)
    hC.set_visible(False)
    plt.show()
    
    
    
    
    





#########################real inf pos 2








nbType = 1

####REAL-INF
fig = plt.figure()
ax = plt.axes()
plt.hold(True)

for i in range(5):
    # i-th boxplot
    bp = plt.boxplot(all_dist_real_inf[1,:,i],(1,0), positions = [ i+1 + i*nbType], widths = 0.8,patch_artist = True)
    plt.setp(bp['boxes'][0], color='red')
    plt.setp(bp['medians'][0], color='white')
i=5
bp = plt.boxplot(all_dist_real_inf[1,:,5], positions = [ i+1 + i*nbType], widths = 0.8,patch_artist = True)# [ 11+60], widths = 0.8,patch_artist = True)
plt.setp(bp['boxes'][0], color='red')
plt.setp(bp['medians'][0], color='white')
# set axes limits and labels
plt.xlim(0,5*nbType-1)
plt.ylim(0,0.05)
plt.title('Error distance between the real trajectory and the infered one' , fontsize=20)
ax.set_xticklabels(labelsName, fontsize=20)
ax.yaxis.label.set_size(18)

midlle= 1
ax.set_xticks([midlle,midlle + (1 + nbType), midlle + (1 + nbType)*2, midlle + (1 + nbType)*3, midlle + (1 + nbType)*4,midlle + (1 + nbType)*5,3 + (nbType)*6+3 ])
ax.yaxis.set_tick_params(labelsize=20)

ax.set_xlabel('Observed data [%]', fontsize=20)
ax.set_ylabel('Distance error [m]', fontsize=20)
# draw temporary red and blue lines and use them to create a legend

hR, = plt.plot([1,1],'r-')
hC, = plt.plot([1,1],'teal')
plt.legend((hR,hC),( '5','69'), title='Latent Space dimension', fontsize=18)

hR.set_visible(False)
hC.set_visible(False)
plt.show()

####RECONSTR-INF

fig = plt.figure()
ax = plt.axes()
plt.hold(True)

for i in range(nbPercent):
    # i-th boxplot
    bp = plt.boxplot(all_dist_reconstr_inf[1,:,i],(1,0), positions =[ i+1 + i*nbType], widths = 0.8,patch_artist = True)
    plt.setp(bp['boxes'][0], color='red')
    plt.setp(bp['medians'][0], color='white')

# set axes limits and labels
plt.xlim(0,5*nbType)
plt.ylim(0,0.06)
plt.title('Error distance between the reconstruction of\n the real trajectory and the infered one' , fontsize=20)
middle = 1
ax.set_xticklabels(labelsName[0:nbPercent], fontsize=20)
ax.set_xticks([midlle,midlle + (1 + nbType), midlle + (1 + nbType)*2, midlle + (1 + nbType)*3, midlle + (1 + nbType)*4,3 + (nbType)*5 + 2 ])
ax.yaxis.set_tick_params(labelsize=20)

ax.set_xlabel('Observed data [%]', fontsize=20)
ax.set_ylabel('Distance error [m]', fontsize=20)


# draw temporary red and blue lines and use them to create a legendhB, = plt.plot([1,1],'b-')
hR, = plt.plot([1,1],'r-')

hC, = plt.plot([1,1],'teal')
plt.legend((hR,hC),('5','69'), title='Latent Space dimension', fontsize=12)

hR.set_visible(False)
hC.set_visible(False)

#plt.savefig('boxcompare.png')
plt.show()
























################################
if allComparison:
    ####REAL-INF
    fig = plt.figure()
    ax = plt.axes()
    plt.hold(True)
    
    for i in range(nbPercent+1):
        # i-th boxplot
        bp = plt.boxplot(np.transpose(all_dist_real_inf[:,:,i],(1,0)), positions = [ i+1 + i*nbType, i+2+i*nbType, i+3+i*nbType,i+4+i*nbType,i+5+i*nbType, i+(i+1)*nbType], widths = 0.8,patch_artist = True)
        setBoxColors(bp)
    bp = plt.boxplot(np.transpose(all_dist_real_inf[:,:,10],(1,0)), positions = [ 11+60, 12+10*nbType, 13+10*nbType,14+10*nbType,15+10*nbType, 16+10*nbType], widths = 0.8,patch_artist = True)# [ 11+60], widths = 0.8,patch_artist = True)
    setBoxColors(bp)
    
    # set axes limits and labels
    plt.xlim(0,5*nbType-1)
    plt.ylim(0,0.075)
    plt.title('Error distance between the real trajectory and the infered one' , fontsize=20)
    ax.set_xticklabels(labelsName, fontsize=20)
    ax.yaxis.label.set_size(18)
    
    midlle= 3
    ax.set_xticks([midlle,midlle + (1 + nbType), midlle + (1 + nbType)*2, midlle + (1 + nbType)*3, midlle + (1 + nbType)*4,midlle + (1 + nbType)*5,3 + (nbType)*6+3 ])
    ax.yaxis.set_tick_params(labelsize=20)
    
    ax.set_xlabel('Observed data [%]', fontsize=20)
    ax.set_ylabel('Distance error [m]', fontsize=20)
    # draw temporary red and blue lines and use them to create a legend
    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    hG, = plt.plot([1,1],'g-')
    hK, = plt.plot([1,1],'k-')
    hM, = plt.plot([1,1],'m-')
    hC, = plt.plot([1,1],'teal')
    plt.legend((hB, hR, hG, hK, hM,hC),('2', '5', '10', '15', '30','69'), title='Latent Space dimension', fontsize=12)
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)
    hK.set_visible(False)
    hM.set_visible(False)
    hC.set_visible(False)
    plt.show()
    
    ####RECONSTR-INF
    
    fig = plt.figure()
    ax = plt.axes()
    plt.hold(True)
    
    for i in range(nbPercent):
        # i-th boxplot
        bp = plt.boxplot(np.transpose(all_dist_reconstr_inf[:,:,i],(1,0)), positions =[ i+1 + i*nbType, i+2+i*nbType, i+3+i*nbType,i+4+i*nbType,i+5+i*nbType, i+(i+1)*nbType], widths = 0.8,patch_artist = True)
        setBoxColors(bp)
    
    
    # set axes limits and labels
    plt.xlim(0,5*nbType-1)
    plt.ylim(0,0.06)
    plt.title('Error distance between the reconstruction of\n the real trajectory and the infered one' , fontsize=20)
    
    ax.set_xticklabels(labelsName[0:nbPercent], fontsize=20)
    ax.set_xticks([midlle,midlle + (1 + nbType), midlle + (1 + nbType)*2, midlle + (1 + nbType)*3, midlle + (1 + nbType)*4,3 + (nbType)*5+3 ])
    ax.yaxis.set_tick_params(labelsize=20)
    
    ax.set_xlabel('Observed data [%]', fontsize=20)
    ax.set_ylabel('Distance error [m]', fontsize=20)
    
    
    # draw temporary red and blue lines and use them to create a legend
    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    hG, = plt.plot([1,1],'g-')
    hK, = plt.plot([1,1],'k-')
    hM, = plt.plot([1,1],'m-')
    hC, = plt.plot([1,1],'teal')
    plt.legend((hB, hR, hG, hK, hM,hC),('2', '5', '10', '15', '30','69'), title='Latent Space dimension', fontsize=12)
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)
    hK.set_visible(False)
    hM.set_visible(False)
    hC.set_visible(False)
    
    #plt.savefig('boxcompare.png')
    plt.show()
    



