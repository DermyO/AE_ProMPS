import numpy as np
import tensorflow as tf
import pdb
import pickle
import Data_driver
import matplotlib.pyplot as plt
import input_data
import VAE

np.random.seed(0)
tf.set_random_seed(0)



#si on veut recup apprentissage précédent
flag_restore = True
flag_restore_data =True
flag_frameMod = False
n_z = 2
training_epochs = 10
batch_size =100

############### Copie données xsens dans structure MNIST ###############
DATA_PARAMS = {}
DATA_PARAMS.update({"data_source": "MVNX", "data_types": ["jointAngle"],"unit_bounds": True})
data_driver = Data_driver.Data_driver(DATA_PARAMS)
if(flag_restore_data==False):    
    data_driver.parse(frameMod=flag_frameMod)
    test = []
    test_data = data_driver.data[:,9,:,:,:]
    test_labels = data_driver.data_labels[::10]
    test.append(test_data)
    test.append(test_labels)
    data_driver.data = data_driver.data.reshape(data_driver.data.shape[0],data_driver.data.shape[1]*data_driver.data.shape[2],data_driver.data.shape[3],data_driver.data.shape[4]) # 1exemple =1 frame = position statique
    
    data_driver.nb_frames = None #1
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
    my_data.train._images = np.zeros([my_data.train._num_examples, data_driver.data.shape[2], data_driver.data.shape[3]],np.float32)
    my_data.train._index_in_epoch= 0
    my_data.train._epoch_completed= 0
    #choices = {'bent_fw': 0, 'bent_fw_strongly': 1,'kicking': 2,'lifting_box': 3,'standing': 4,'walking': 5,'window_open': 6}
    choices = {'Setup_A_Seq_1': 0, 'Setup_A_Seq_2': 1,'Setup_A_Seq_3': 2,'Setup_A_Seq_4': 3,'Setup_A_Seq_5': 4,'Setup_A_Seq_6': 5}
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
    my_data.validation._images = np.zeros([my_data.validation._num_examples, data_driver.data.shape[2], data_driver.data.shape[3]],np.float32)
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
    my_data.test._images = np.zeros([my_data.test._num_examples, data_driver.data.shape[2], data_driver.data.shape[3]],np.float32)
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
    data_driver.save_data("./data/save_dataDriver_xpAdrienPos",0,data_driver)#adapt to record data_driver
    data_driver.save_data("./data/save_my_data_xpAdrienPos",0,my_data)
    data_driver.save_data("./data/save_test_xpAdrienPos",0,test)
    pdb.set_trace()

else:
    mnist = data_driver.read_data("./data/save_my_data_xpAdrienPos",0)
    data_driver = data_driver.read_data("./data/save_dataDriver_xpAdrienPos",0)
    test_data, test_labels =  data_driver.read_data("./data/save_test_xpAdrienPos",0)
x_len = 22
y_len = 3
############### Fin copie ############################################################
#création d'un espace latent de dimension n_z
with tf.name_scope('network_archi'):
    network_architecture = dict(n_hidden_recog_1=500, # 1st layer encoder neurons
             n_hidden_recog_2=500, # 2nd layer encoder neurons
             n_hidden_gener_1=500, # 1st layer decoder neurons
             n_hidden_gener_2=500, # 2nd layer decoder neurons
             n_input=x_len*y_len,#784, # MNIST data input (img shape: 28*28)
             n_z=n_z)#20)  # dimensionality of latent space

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

with tf.name_scope('VAE'):
	vae = train(n_z, network_architecture,restore=flag_restore, training_epochs=training_epochs,batch_size=batch_size, x_len=x_len,y_len=y_len)
if(flag_restore==False):
    vae.save_session(training_epochs,n_z,batch_size)



#####Test reconstr
x_sample2 = test_data[0] #premiere traj premier typex
tmp_val = x_sample2.shape
x_sample2 = x_sample2.reshape(tmp_val[0],tmp_val[1]*tmp_val[2])
 
#TODO plot la figure

#pour correspondre a la fonction de maxime

x_reconstr_names = test_labels
x_sample2 = x_sample2[0:50600,:]
x_sample2 = [x_sample2]
x_sample2 = np.array(x_sample2)

#x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
DATA_VISUALIZATION = {}
#            "data_source": source,
#            "data_types": data_types,
#            "as_3D": as_3D,
#            "unit_bounds": True
pdb.set_trace()
DATA_VISUALIZATION.update({
    'transform_with_all_vtsfe': False,
    'dynamic_plot': False,
    'body_lines': True,
    'data_inf': [], 
    'show': True, 
    'nb_samples_per_mov': 1,
    'average_reconstruction': False,
    "data_driver" : data_driver,
    "reconstr_datasets" : x_reconstr2, #(1, 1, 70, 70, 66)
    "reconstr_datasets_names" : x_reconstr_names, #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
    "x_samples" : x_sample2, #(70,70,66)
    "sample_indices" : [0], #8
    "only_hard_joints" : False,
    "mov_types" : [data_driver.mov_types[0]],
    "displayed_movs" :[data_driver.mov_types[0]],
    #"data_inf" : data_inf,
    "plot_3D" : True, #add ori
    "time_step" : 100
    })
vae.show_data(**DATA_VISUALIZATION)
plt.show()
