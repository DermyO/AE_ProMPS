from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import pdb
import pickle
import Data_driver
import matplotlib.pyplot as plt
import input_data
import matplotlib.cm as cm
import time
from matplotlib import animation
import AE
from connector import Connector
from dataTreatment import DataTreatment
from my_statistics import My_statistics
#si on veut recup apprentissage précédent
flag_restore = True
flag_restore_data =True
flag_frameMod = False
flag_plot_image = True
flag_plot_squeleton = True
flag_save_ls = False
commWithMatlab = False
launch_stats = False
batch_size =1000
training_epochs = 5000
learning_rate = 0.001
display_step = 1000
examples_to_show = 10
typeActivation = 'leaky' #sigmoid apprend mal
typeInit = 'xavier_init' #random apprend mal
typeOpti='losses.mean_squared_error'
x_len = 23
y_len = 3
# Network Parameters
num_hidden_1 = 500#256 # 1st layer num features
n_z = 5
num_input = x_len*y_len # MNIST data input (img shape: 28*28)


############### Copie données xsens dans structure MNIST ###############
np.random.seed(0)
tf.set_random_seed(0)
DATA_PARAMS = {}
DATA_PARAMS.update({"data_source": "MVNX", 'as_3D': True, 'data_types': ['position'],"unit_bounds": False})
data_driver = Data_driver.Data_driver(DATA_PARAMS)
if(flag_restore_data==False):    
    data_driver.parse(frameMod=flag_frameMod)
    test = []
    test_data = data_driver.data[:,9,:,:]
    test_labels = data_driver.data_labels[::10]
    test.append(test_data)
    test.append(test_labels)
    data_driver.data = data_driver.data.reshape(data_driver.data.shape[0],data_driver.data.shape[1]*data_driver.data.shape[2],data_driver.data.shape[3]) # 1exemple =1 frame = position statique
    #data_driver.nb_frames = None #1
    data_driver.data_labels = None
    #todo: améliorer : ici on fait ça pour avoir lla meme structure que les données MNIST
    my_data = input_data.read_data_sets('MNIST_data', one_hot=True) 
    val_tmp = int(data_driver.data.shape[1] - 2*data_driver.data.shape[1]/10 ) #2/10e sont conservés pour les tests
    val_tmp2 = int(data_driver.data.shape[1] - data_driver.data.shape[1]/10 )
    tall_tmp = data_driver.data.shape[1]/10

    #training Data
    my_data.train._images = None
    my_data.train._num_examples = data_driver.data.shape[0]*val_tmp
    my_data.train._labels = np.zeros([my_data.train._num_examples,data_driver.data.shape[0]], np.float32)
    my_data.train._images = np.zeros([my_data.train._num_examples, data_driver.data.shape[2]],np.float32)
    my_data.train._index_in_epoch= 0
    my_data.train._epoch_completed= 0
    choices = {'bent_fw': 0, 'bent_fw_strongly': 1,'kicking': 2,'lifting_box': 3,'standing': 4,'walking': 5,'window_open': 6}
    #choices = {'Setup_A_Seq_1': 0, 'Setup_A_Seq_2': 1,'Setup_A_Seq_3': 2,'Setup_A_Seq_4': 3,'Setup_A_Seq_5': 4,'Setup_A_Seq_6': 5}
    idx=0
    for i in range(data_driver.data.shape[0]): #chaque type 
        for j in range(val_tmp): #pour chaque exemple d'entrainement du type i
            #k = i*data_driver.data.shape[1] + j   
            #result = choices.get(data_driver.data_labels[k], 'default')
            my_data.train._labels[idx][i] = 1
            my_data.train._images[idx] = data_driver.data[i][j]
            idx = idx+1
    
    #validation Data
    my_data.validation._images = None
    my_data.validation._num_examples = int(data_driver.data.shape[0]*tall_tmp) 
    my_data.validation._labels = np.zeros([my_data.validation._num_examples,data_driver.data.shape[0]], np.float32)
    my_data.validation._images = np.zeros([my_data.validation._num_examples, data_driver.data.shape[2]],np.float32)
    my_data.validation._index_in_epoch= 0
    my_data.validation._epoch_completed= 0
    idx=0
    for i in range(data_driver.data.shape[0]):
        for j in range(val_tmp,val_tmp2):
        #k = i*data_driver.data.shape[1] + j   
        #result = choices.get(data_driver.data_labels[k], 'default')
            my_data.validation._labels[idx][i] = 1
            my_data.validation._images[idx] = data_driver.data[i][j]
            idx = idx+1
        
    #test Data
    my_data.test._images = None
    my_data.test._num_examples =  int(data_driver.data.shape[0]*tall_tmp)
    my_data.test._labels = np.zeros([my_data.test._num_examples,data_driver.data.shape[0]], np.float32)
    my_data.test._images = np.zeros([my_data.test._num_examples, data_driver.data.shape[2]],np.float32)
    my_data.test._index_in_epoch= 0
    my_data.test._epoch_completed= 0
    idx=0
    for i in range(data_driver.data.shape[0]):
        for j in range(val_tmp2, data_driver.data.shape[1]):
        #k = i*data_driver.data.shape[1] + j   
        #result = choices.get(data_driver.data_labels[k], 'default')
            my_data.test._labels[idx][i] = 1
            my_data.test._images[idx] = data_driver.data[i][j]
            idx = idx+1
    
    mnist = my_data
   
    data_driver.save_data("./data/save_dataDriver_actionPos_without_norm",0,data_driver)#adapt to record data_driver
    data_driver.save_data("./data/save_my_data_actionPos_without_norm",0,my_data)
    data_driver.save_data("./data/save_test_actionPos_without_norm",0,test)


else:
    mnist = data_driver.read_data("./data/save_my_data_actionPos_without_norm",0)#_xpAdrienPos",0)
    data_driver = data_driver.read_data("./data/save_dataDriver_actionPos_without_norm",0) # _xpAdrienPos",0)
    test_data, test_labels =  data_driver.read_data("./data/save_test_actionPos_without_norm",0)#_xpAdrienPos",0)

############### Fin copie ############################################################




ae = AE.AE(num_input,num_hidden_1,n_z,learning_rate,typeInit = typeInit, typeActivation=typeActivation,typeOpti=typeOpti,restore=flag_restore, epoch=training_epochs)
dt = DataTreatment

if(flag_restore == False):    # If we don't restore a previous learning, then we train
    for i in range(1, training_epochs+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
    
        # Run optimization op (backprop) and cost op (to get loss value)
        l = ae.partial_fit(batch_x, i)

        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
    ae.save_session(training_epochs,n_z)


if(flag_plot_image or flag_plot_squeleton): #unitary test to verify if the reconstruction learning is ok
    n = examples_to_show*examples_to_show
    canvas_orig = np.empty((x_len * n, y_len * n))
    canvas_recon = np.empty((x_len * n, y_len * n))
    #for i in range(n):
    i=1
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # Encode and decode the digit image
    g = ae.sess.run(ae.decoder_op, feed_dict={ae.X: batch_x})
    # Display original images
    for j in range(n):
        # Draw the original digits
        canvas_orig[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len] = \
            batch_x[j].reshape([x_len, y_len])
    # Display reconstructed images
    for j in range(n):
        # Draw the reconstructed digits
        canvas_recon[i * x_len:(i + 1) * x_len, j * y_len:(j + 1) * y_len] = \
            g[j].reshape([x_len, y_len])

    #ploting Image
    if(flag_plot_image==True):    
        print("Original and Reconstructed Images")
        plt.figure(figsize=(n, n))
        plt.subplot(1,2,1)
        
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        # plt.show()
        plt.subplot(1,2,2)
        #print("Reconstructed Images")
        #plt.figure(figsize=(n, n))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.show()


    #ploting squeleton
    if (flag_plot_squeleton==True):
        g2 = g
        DATA_PARAMS = {}
        DATA_PARAMS.update({"data_source": "MVNX", 'as_3D': True, 'data_types': ['position'],"unit_bounds": True})
        data_driver = Data_driver.Data_driver(DATA_PARAMS)
        data_driver.as_3D = True
        DATA_VISUALIZATION = {}
        DATA_VISUALIZATION.update({
            'transform_with_all_vtsfe': False,
            'dynamic_plot': False,
            'body_lines': True,
            'data_inf': [], 
            'show': True, 
            'nb_samples_per_mov': 1,
            'average_reconstruction': False,
            "data_driver" : data_driver,
            "reconstr_datasets" : g2, #(1, 1, 70, 70, 66)
            "reconstr_datasets_names" : ['testWithAE'], #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
            "x_samples" : batch_x, #(70,70,66)
            "sample_indices" : [0], #8
            "only_hard_joints" : False,
           # "mov_types" : ['bent_fw'],
           "displayed_movs" :['random posture'],
            #"data_inf" : data_inf,
            "plot_3D" : True, #add ori
            "time_step" : 1,
            'n_z': n_z,
            #"as_3D" : Trueki
            })
        dt.show_data(**DATA_VISUALIZATION)
            


if flag_save_ls : #to save LS of all trajectory (to allow matlab to learn ProMPs)
    nbAction = data_driver.data.shape[0]
    nbFrames = data_driver.nb_frames
    z = np.zeros([nbAction,10,nbFrames,n_z],np.float32)
    for i in range(nbAction):
        for j in range(10):
            batch_x = data_driver.data[i,nbFrames*j:1325*(j+1),:]
            # Encode and decode the digit image
            z[i,j] = ae.retrieve_latent_space(batch_x)
    dt.saveAllLS(z,nbLS=n_z,nbFrame=nbFrames)


if commWithMatlab:
    
    connex = Connector() #YARP connexion with matlab
    list_zs = []
 #   time_start = time.clock()
    for i in range(7):
        connex.addMessage("ask_data")            
        zs = connex.readFloat(nbData = [70,n_z]) #read latent space that comes from Matlab
        list_zs.append(zs) 
    connex.closeConnector() #YARP disconnexion
#    time_retrieve_data = (time.clock() - time_start) / 7    
    #retrieve inf traj
    x_inf = np.zeros([7,70,num_input])
 #   time_inf = np.zeros(7,np.float32)
    for i in range(7):
        time_start = time.clock()
        x_inf[i] =  ae.sess.run(ae.decoder_op, feed_dict={ae.encoder_op: list_zs[i]})
 #       time_inf[i] = (time.clock() - time_start)
    #retrieve real and reconstr trajectories
    x_reconstr = np.zeros([7,70,num_input],np.float32)
    x_real = np.zeros([7,70,num_input],np.float32)
    for i in range(7):
        x_real[i] = dt.retrieveRealTest(data_driver.data,numTest=9,numAction=i,nbFrame=1325)
        x_reconstr[i] = ae.sess.run(ae.decoder_op, feed_dict={ae.X: x_real[i]})

    x_real.reshape(7,70,x_len, y_len)
    x_reconstr.reshape(7,70,x_len, y_len)
    x_inf.reshape(7,70,x_len, y_len)
   
    #ploting squeleton
    DATA_PARAMS = {}
    DATA_PARAMS.update({"data_source": "MVNX", 'as_3D': True, 'data_types': ['position'],"unit_bounds": True})
    data_driver = Data_driver.Data_driver(DATA_PARAMS)
    data_driver.as_3D = True
    DATA_VISUALIZATION = {}
    for i in range(7):
        DATA_VISUALIZATION.update({
            'transform_with_all_vtsfe': False,
            'dynamic_plot': False,
            'body_lines': True,
            'data_inf': x_inf[i], 
            'show': True, 
            'nb_samples_per_mov': 1,
            'average_reconstruction': False,
            "data_driver" : data_driver,
            "reconstr_datasets" : x_reconstr[i], #(1, 1, 70, 70, 66)
            "reconstr_datasets_names" : [data_driver.mov_types[i]], #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
            "x_samples" : x_real[i], #(70,70,66)
            "sample_indices" : [0], #8
            "only_hard_joints" : False,
           # "mov_types" : [data_driver.mov_types[i]],
           "displayed_movs" :[data_driver.mov_types[i]],
            "plot_3D" : True, #add ori
            "time_step" : 1,
            'n_z': n_z,
            'percent': 60
            })
        dt.show_data(**DATA_VISUALIZATION)
 
  

if launch_stats:
    nbStat = 5

    #retrieve from matlab and save into file
    #connex = Connector() #YARP connexion with matlab
    nbTrial = 10
    global_list = np.zeros([nbStat,7*nbTrial,70,n_z])
    list_error = np.zeros([nbStat,70])
    perc=0
    #saved in 'AE_befor_stats_ls_'+str(n_z)
    for nbPercent in range(nbStat):
        perc=perc+1
        print('nbPercent: ', str(nbPercent))
        list_zs = []
        for mov in range(1,8):
            print('mov: ', str(mov))
            for trial in range(1,nbTrial+1):
                print('test: ', str(trial))
                nameFile = './data/inference_ZS_from_matlab/'+ 'LS_'+str(n_z)+'-perc_'+str(perc)+'-mov_'+str(mov)+'-trial_'+str(trial)+'.txt'
                print(nameFile)
                cpt=0
                zs = np.zeros([70,n_z],np.float32)
                with open(nameFile, 'r') as f:
                    lines = [line.strip('\n') for line in f.readlines()]
                    for line in lines:
                        zs[cpt,:] = np.asarray(line.split())                            
                        cpt=cpt+1
#                connex.addMessage("ask_data")            
 #               zs = connex.readFloat(nbData = [70,n_z]) #read latent space that comes from Matlab

                list_zs.append(zs) 
        #print('ask_error_list')
        #connex.addMessage("ask_error_list")
        #list_error[nbPercent,:] = connex.readFloat(nbData = [70,1],flag_debug =True);
        nameFile = './data/inference_ZS_from_matlab/'+ 'list_error_LS_'+str(n_z)+'-perc_'+str(perc)+'.txt'
        print(nameFile)
        
        with open(nameFile, 'r') as f:
            lines = [line.strip('\n') for line in f.readlines()]
            for line in lines:
                list_error[nbPercent,:]  = np.asarray(line.split())                                        
        global_list[nbPercent] = list_zs
    with open('AE_befor_stats_ls_'+str(n_z), 'wb') as fichier:
        mon_dep = pickle.Pickler(fichier)
        mon_dep.dump([global_list,list_error])
   #print("close matlab connexion")
    #connex.closeConnector() #YARP disconnexion

    
    #retrieve from file and stats
    with open('AE_befor_stats_ls_'+str(n_z), 'rb') as fichier:
        mon_dep = pickle.Unpickler(fichier)
        [global_list, list_error] = mon_dep.load()

    #retrieve inf traj
    x_inf = np.zeros([nbStat,7*nbTrial,70,num_input])
    for nbPercent in range(nbStat):
        for i in range(7*nbTrial):
            tmp_val =  ae.sess.run(ae.decoder_op, feed_dict={ae.encoder_op: global_list[nbPercent,i]})
            x_inf[nbPercent,i] = tmp_val

    #retrieve real and reconstr trajectories
    x_reconstr = np.zeros([7*nbTrial,70,num_input],np.float32)
    x_real = np.zeros([7*nbTrial,70,num_input],np.float32)
    for i in range(7):
        for j in range(nbTrial):
            x_real[nbTrial*i + j] = dt.retrieveRealTest(data_driver.data,numTest=j+1,numAction=i,nbFrame=1325)
            x_reconstr[nbTrial*i + j] = ae.sess.run(ae.decoder_op, feed_dict={ae.X: x_real[nbTrial*i + j]})

    x_real_plot = x_real.reshape(7*nbTrial,70,x_len, y_len)
    x_reconstr_plot = x_reconstr.reshape(7*nbTrial,70,x_len, y_len)
    x_inf_plot = x_inf.reshape(nbStat,7*nbTrial,70,x_len, y_len)
    
    my_statistics = My_statistics(x_real, x_inf, x_reconstr, [nbStat,7*nbTrial,70,69]) #TODO rendre 5/10 en variable

    with open('AE_myLittleStats_LS_'+str(n_z)+'_epochs_'+str(training_epochs), 'wb') as fichier:
        mon_pickler= pickle.Pickler(fichier)
        mon_pickler.dump(my_statistics)

