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


#entrainement des données
def train(n_z, network_architecture,restore=False,  learning_rate=0.001, batch_size=100, training_epochs=10, display_step=5, x_len=28,y_len=28):
    vae = VAE.VAE(network_architecture, n_z, learning_rate=learning_rate, batch_size=batch_size,save_path="./test_save_path/",restore=restore, epoch=5, x_len=x_len, y_len=y_len)
    # Training cycle
    if(restore == False):
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
    
            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)
                
                batch_xs = batch_xs.reshape(batch_xs.shape[0],batch_xs.shape[1])
                # Fit training using batch data
                cost = vae.partial_fit(batch_xs, epoch)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size
    
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), 
                      "cost=", "{:.9f}".format(avg_cost))
    return vae



#si on veut recup apprentissage précédent
flag_restore = True
flag_restore_data =True
flag_frameMod = False
n_z = 5
training_epochs = 10
batch_size =1

############### Copie données xsens dans structure MNIST ###############
DATA_PARAMS = {}
DATA_PARAMS.update({"data_source": "MVNX", 'as_3D': True, 'data_types': ['position'],"unit_bounds": True})
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
    data_driver.save_data("./data/save_dataDriver_actionPos",0,data_driver)#adapt to record data_driver
    data_driver.save_data("./data/save_my_data_actionPos",0,my_data)
    data_driver.save_data("./data/save_test_actionPos",0,test)


else:
    mnist = data_driver.read_data("./data/save_my_data_actionPos",0)#_xpAdrienPos",0)
    data_driver = data_driver.read_data("./data/save_dataDriver_actionPos",0) # _xpAdrienPos",0)
    test_data, test_labels =  data_driver.read_data("./data/save_test_actionPos",0)#_xpAdrienPos",0)
x_len = 23
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

#if(n_z ==2):
    #x_sample, y_sample = mnist.test.next_batch(batch_size) 
    #vall = x_sample.shape
    #x_sample = x_sample.reshape(vall[0],vall[1]*vall[2])
    #z_mu = vae.transform(x_sample)
    #plt.figure(figsize=(8, 6)) 
    #plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    #plt.colorbar()
    #plt.grid()
    
    #nx = ny = 20
    #x_values = np.linspace(-3, 3, nx)
    #y_values = np.linspace(-3, 3, ny)
    
    #canvas = np.empty((x_len*ny, y_len*nx))
    #for i, yi in enumerate(x_values):
        #for j, xi in enumerate(y_values):
            #z_mu = np.array([[xi, yi]]*vae.batch_size)
            #x_mean = vae.generate(z_mu)
            #canvas[(nx-i-1)*x_len:(nx-i)*x_len, j*y_len:(j+1)*y_len] = x_mean[0].reshape(x_len, y_len)
    
    #plt.figure(figsize=(8, 10))        
    #Xi, Yi = np.meshgrid(x_values, y_values)
    #plt.imshow(canvas, origin="upper", cmap="gray")
    #plt.tight_layout()
    #plt.show()


##ici on récupère quelques tests (taille = 100 * 784)
#with tf.name_scope('x_sample'):
    #x_sample = mnist.test.next_batch(batch_size)[0]

##on utilise l'encodeur et le décodeur dessus
#with tf.name_scope('x_reconstruct'):
    #x_reconstruct = vae.reconstruct(x_sample)

#nbDrawing =4
#plt.figure(figsize=(8, 12))
#for i in range(nbDrawing):
    #tf.summary.image("x_sample_"+str(i),x_sample[i].reshape(x_len, y_len))
    #tf.summary.image("x_reconstruct_"+str(i),x_reconstruct[i].reshape(x_len, y_len))
    #plt.subplot(nbDrawing, 2, 2*i + 1)
    #plt.imshow(x_sample[i].reshape(x_len, y_len), vmin=0, vmax=1, cmap="gray")
    #plt.title("Test input")
    #plt.colorbar()
    #plt.subplot(nbDrawing, 2, 2*i + 2)
    #plt.imshow(x_reconstruct[i].reshape(x_len, y_len), vmin=0, vmax=1, cmap="gray")
    #plt.title("Reconstruction")
    #plt.colorbar()
#plt.show()



#####Test reconstr
x_sample2 = test_data[0] #premiere traj premier typex

tmp_val = int(x_sample2.shape[0]/batch_size)
x_reconstr2 = np.zeros([tmp_val*batch_size, x_sample2.shape[1]], np.float32)
for i in range(tmp_val):
    #x_reconstruct2[i*100:(i+1)*100,:] = 
    x_reconstr2[i*batch_size:(i+1)*batch_size,:]  =   vae.reconstruct(x_sample2[i*batch_size:(i+1)*batch_size])
    
#TODO plot la figure
pdb.set_trace
#pour correspondre a la fonction de maxime
x_reconstr2 = [[[x_reconstr2]]]
x_reconstr2 = np.array(x_reconstr2)
x_reconstr_names = test_labels
#x_sample2 = x_sample2[0:50600,:]
x_sample2 = [x_sample2]
x_sample2 = np.array(x_sample2)

##TODO test enlever apres
#x_reconstr2 = x_sample2
#x_reconstr2 = [[x_reconstr2]]
#x_reconstr2 = np.array(x_reconstr2)

#x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
DATA_VISUALIZATION = {}
#            "data_source": source,
#            "data_types": data_types,
#            "as_3D": as_3D,
#            "unit_bounds": True

data_driver.as_3D = True
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
    "time_step" : 5,
    #"as_3D" : True
    })
vae.show_data(**DATA_VISUALIZATION)
