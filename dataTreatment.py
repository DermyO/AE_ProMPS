import pdb
import numpy as np
import math
import tensorflow as tf
import pickle
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
import sys
       
class DataTreatment(object):
    """All data treatment (plot, record,analyse)"""
        
    
    def saveAllLS(data,nbLS=2,nbFrame=1325):
    
        sampling_step = int(nbFrame / 70) 
        
        for actionType in range(7):
            try:
                os.mkdir("./data_ae_"+str(actionType))
            except OSError:
                print("Error during creation: "+"./data_ae_"+str(actionType))
                pass
            for innx in range(10):
                f = open("./data_ae_"+str(actionType)+"/record"+str(innx)+".txt", "w+")
                for vb in range(0,sampling_step*70,sampling_step):
                    nameString = ''
                    for nbstring in range(nbLS-1):
                        nameString += str(data[actionType,innx,vb,nbstring])+"\t"
                    nameString +=str(data[actionType,innx,vb,nbLS-1])+"\n"
                    f.write(nameString)
                f.close()
        
    def retrieveRealTest(data,numTest,numAction,nbFrame=1325):
    
        data2 = np.zeros([70,data.shape[2]],np.float32)
        innx = numTest
        subData =data[numAction,nbFrame*(numTest-1):nbFrame*(numTest),:]
        sampling_step = int(nbFrame / 70) 
        cpt=0
        for vb in range(0,sampling_step*70,sampling_step):
                data2[cpt,:] = subData[vb,:]
                cpt+=1
        return data2
    
    def show_data(data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices, x_samples, n_z = -10, percent = -10, plot_variance=False, nb_samples_per_mov=1, show=True, displayed_movs=None, dynamic_plot=False, plot_3D=False, window_size=10, time_step=5, body_lines=True, only_hard_joints=True, transform_with_all_vtsfe=True, average_reconstruction=True, data_inf=[]):
        """
            Data space representation.
            You can 2optionally plot reconstructed data as well at the same time.
        """
        
        nb_frames = len(x_samples)
        # labels = data_driver.data_labels
        
        display_per_mov = 1
        
        nb_sub_sequences = 1
        
        nb_colors = 3#display_per_mov +1
        colors = cm.rainbow(np.linspace(0, 1, nb_colors))
        
        # to avoid modifying x_samples
        data = np.copy(x_samples) 
        data = data.reshape([ nb_frames, -1, 3])
        
        if(data_inf != []): 
            tmp_shape = data_inf.shape
            data_inf = data_inf.reshape(nb_frames,23,3)
                
        data_reconstr = []
        data_reconstr.append(reconstr_datasets.reshape([nb_frames, -1, 3]))
        segment_count = len(data[0])#22
        
        cs = []
        
        #if not body_lines:
            # color every segment point, a color per sample
            #for i in range(display_per_mov):
                #for j in range(segment_count):
                    #cs.append(colors[i])
        
        # plots = []
        # plot_recs = []
        
        body_lines_indices = [[0, 7], [7, 11], [11, 15], [15, 19], [19, 23]]
        additional_lines = [[15, 0, 19], [7, 11]]
        nb_body_lines = len(body_lines_indices)
        nb_additional_lines = len(additional_lines)
        
        if body_lines:
            def plot_body_lines(plots, k, data):
                for i in range(nb_body_lines):
                    line_length = body_lines_indices[i][1] - body_lines_indices[i][0]
                    # NOTE: there is no .set_data() for 3 dim data...
                    # plot 2D
                    plots[i].set_data(
                        data[
                            k,
                            body_lines_indices[i][0] : body_lines_indices[i][1],
                            0
                        ],
                        data[
                            k,
                            body_lines_indices[i][0] : body_lines_indices[i][1],
                            1
                        ]
                    )
                    # plot the 3rd dimension
                    plots[i].set_3d_properties(
                        data[
                            k,
                            body_lines_indices[i][0] : body_lines_indices[i][1],
                            2
                        ]
                    )
        
                for i in range(nb_additional_lines):
                    # additional_lines_data shape = [display_per_mov, nb_additional_lines, 3, line_length, nb_frames]
        
                    # plot 2D
                    plots[nb_body_lines+i].set_data(
                        data[
                            k,
                            additional_lines[i],
                            0
                        ],
                        data[
                            k,
                            additional_lines[i],
                            1
                        ]
                    )
                    # plot the 3rd dimension
                    plots[nb_body_lines+i].set_3d_properties(
                        data[
                            k,
                            additional_lines[i],
                            2
                        ]
                    )
        
            def animate(k):
                index = 0
                step = nb_body_lines + nb_additional_lines
                next_index = index + step
                plot_body_lines(plots[index : next_index], k, data)
                if(data_inf != []):
                    plot_body_lines(plot_inf[index : next_index], k, data_inf)
        
                for sub in range(nb_sub_sequences):
                    plot_body_lines(plot_recs[sub][index : next_index], k, data_reconstr[sub])
                index = next_index
                title.set_text("Time = {}".format(k))
                if not show:
                    ax.view_init(30, -150 + 0.7 * k)
        #else:
            #def animate(k):
                #indices = [data_driver.mov_indices[mov] + j for j in sample_indices]
                #plots[0]._offsets3d = (
                    #data[indices, k, :, 0].reshape([segment_count*display_per_mov]),
                    #data[indices, k, :, 1].reshape([segment_count*display_per_mov]),
                    #data[indices, k, :, 2].reshape([segment_count*display_per_mov])
                #)
                #for r,reco in enumerate(data_reconstr):
                    #for sub in range(nb_sub_sequences):
                        #plot_recs[r*nb_sub_sequences + sub]._offsets3d = (
                            #reco[sub, indices, k, :, 0].reshape([segment_count*display_per_mov]),
                            #reco[sub, indices, k, :, 1].reshape([segment_count*display_per_mov]),
                            #reco[sub, indices, k, :, 2].reshape([segment_count*display_per_mov])
                        #)
                #title.set_text("Time = {}".format(k))
                #if not show:
                    #ax.view_init(30, -150 + 0.7 * k)
        
        if displayed_movs is not None:
            # turn on interactive mode
            plt.ion()
            # scatter all movements in displayed_movs, a color per movement sample
            for i, mov in enumerate(displayed_movs):
                plots = []# plot data original
                plot_recs = [] # plot data reconstr
                plot_inf = [] # plot data inf
        
                fig = plt.figure(figsize=(8, 6))
                fig.canvas.set_window_title("Input data space - "+mov)
                ax = fig.gca(projection='3d')
                box_s = 1
                ax.grid(False)
                ax.set_xlim3d(-box_s, box_s)
                ax.set_ylim3d(-box_s, box_s)
                ax.set_zlim3d(-box_s, box_s)
                # set point-of-view: specified by (altitude degrees, azimuth degrees)
                ax.view_init(30, -150)
                title = ax.set_title("Time = 0")
                if body_lines:
                    #Prepare bodylines for the x_original plot
                    # plot the nb_body_lines+nb_additional_lines lines of body segments
                    nameLabel = 'Ground truth: ' + mov

                    for k in range(nb_body_lines + nb_additional_lines):
                        # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                        plots.append(ax.plot([], [], [], label=nameLabel, c='k', marker='o')[0])
                        nameLabel = None
        

                    if(data_inf!=[]):
                        #Prepare bodylines for the x inf plot
                        # plot the nb_body_lines+nb_additional_lines lines of body segments
                        nameLabel = 'Inference from ' + str(percent) + '%'
                        for k in range(nb_body_lines + nb_additional_lines):
                            # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                            plot_inf.append(ax.plot([], [], [], label=nameLabel, c='r', marker='D', linestyle='-.')[0])
                            nameLabel = None
        
                    # Prepare bodylines for the x reconstr plot
                    nameLabel = 'Reconstruction with L.S. =' + str(n_z)
                    for sub in range(nb_sub_sequences):
                        plts = []
                        # plot the nb_body_lines + nb_additional_lines lines of body reconstructed segments
                        for k in range(nb_body_lines + nb_additional_lines):
                            if k != 0 or sub != 0:
                                nameLabel = None
                            plts.append(ax.plot([], [], [], label=nameLabel, c='g', marker='D', linestyle='dashed')[0])
                        # plot_recs shape = [nb_reconstr_data*nb_sub_sequences, display_per_mov*(nb_body_lines + nb_additional_lines)]
                        plot_recs.append(plts)
                else:
                    # just scatter every point
                    plots.append(ax.scatter([], [], [], c=cs))
                    label = reconstr_datasets_names[0]
                    plot_recs.append(ax.scatter([], [], [], label=label, c=cs, marker='D'))
        
                plt.legend()
                # call the animator.  blit=True means only re-draw the parts that have changed.
                anim = animation.FuncAnimation(fig, animate, frames=nb_frames, interval=time_step, blit=False, )
                # Save as mp4. This requires mplayer or ffmpeg to be installed
                #record_animation(anim, "-input_data_space")
                if show:
                    plt.show()
                    _ = input("Press [enter] to continue.") # wait for input from the user
                plt.close()    # close the figure to show the next one.        
            # turn off interactive mode
            plt.ioff()
