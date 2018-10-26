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
#from tensorflow.python import debug as tf_debug


def xavier_init(fan_in, fan_out, constant=1): 
        """ Xavier initialization of network weights"""
        # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
        high = constant*np.sqrt(6.0/(fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class VAE(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture,n_z,  transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=100, save_path=None,restore=False, epoch=10, x_len=28, y_len=28):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_path = save_path
        self.restore =restore 
        self.merged_summary = None
        self.epoch = 0
        self.train_writer = None
        self.test_writer = None
        self.x_len=x_len
        self.y_len=y_len

        # tf Graph input
        with tf.name_scope('x'):
                self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
	
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # init session to restore or randomly initialize all variables in network
        self.init_session(epoch, n_z)
    	
	#de VarAutoencod
	# Initializing the tensor flow variables
        #init = tf.global_variables_initializer()
        # Launch the session
        #self.sess = tf.InteractiveSession()
        #self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        with tf.name_scope('Weights'):
                network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(network_weights["weights_recog"], network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        with tf.name_scope('z'):
                self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps), name="z")

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = self._generator_network(network_weights["weights_gener"], network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1,  n_hidden_gener_2, n_input, n_z):
        all_weights = dict()

        with tf.name_scope('w_enc'):
                all_weights['weights_recog'] = {
                    'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1),name="h1"),
                    'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2),name="h2"),
                    'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z),name="out_mean"),
                    'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z),name="out_log_sigma")}
        try:
                tf.summary.histogram("w_enc_h1",all_weights['weights_recog']['h1'])
                tf.summary.histogram("w_enc_h2",all_weights['weights_recog']['h2'])
                tf.summary.histogram("w_enc_out",all_weights['weights_recog']['out_mean'])
        except:
                print("Error  w enc summaries")
        with tf.name_scope('b_enc'):    
                all_weights['biases_recog'] = {
                    'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32),name="b1"),
                    'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32),name="b2"),
                    'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name="out_mean"),
                    'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32),name="out_log_sigma")}
        with tf.name_scope('w_dec'): 
                all_weights['weights_gener'] = {
                    'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1),name="h1"),
                    'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2),name="h2"),
                    'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input),name="out_mean"),
                    'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input),name="out_log_sigma")}
        with tf.name_scope('b_dec'): 
                all_weights['biases_gener'] = {
                    'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32),name="b1"),
                    'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32),name="b2"),
                    'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32),name="out_mean"),
                    'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32),name="out_log_sigma")}
        try:
                tf.summary.histogram("w_rec_h1",all_weights['weights_gener']['h1'])
                tf.summary.histogram("w_rec_h2",all_weights['weights_gener']['h2'])
                tf.summary.histogram("w_rec_out",all_weights['weights_gener']['out_mean'])
        except:
                print("Error w dec summaries")

        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        with tf.name_scope('Encoder'):
                with tf.name_scope('layer_1'):
                        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']), name="out_layer_1")     
                with tf.name_scope('layer_2'):
                        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']),name="out_layer_2") 
                with tf.name_scope('z'):
                        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'], name="z_mean")
                        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'], name="z_log_sig_sq")
        imgTest = tf.reshape(self.x, [-1,self.x_len,self.y_len,1])
        try:
                tf.summary.image("x", imgTest)
        except:
                print("Error x summary")
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        with tf.name_scope('Decoder'):
                with tf.name_scope('layer_1'):
                        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']), name="out_layer_1") 
                with tf.name_scope('layer_2'):
                        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']), name="out_layer_2") 
        with tf.name_scope('x_reco'):
                x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']), name="x_reconstr_mean")
        #print("shape x_reconstr="+str(x_reconstr_mean.shape))
        #print("x_len=", str(self.x_len))
        imgTest = tf.reshape(x_reconstr_mean, [-1,self.x_len,self.y_len,1])
        try:
                tf.summary.image("x_reco", imgTest)
        except:
                 print("Error xreco summary")

        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        with tf.name_scope('reconstr_loss'):
                reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1, name="reconstr_loss" )
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
                latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1, name="latent_loss")
                self.cost = tf.reduce_mean(reconstr_loss + latent_loss, name="our_cost")   # average over batch
     #   if (math.isfinite(tf.to_float(self.cost))):
        tf.summary.histogram("cost", self.cost)    
       # else:
        #        print("Error cost inf")
        #        pdb.set_trace()
               
	# Use ADAM optimizer
        with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X, epoch):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
	
        #opt, cost = self.sess.run((self.optimizer, self.cost),  feed_dict={self.x: X})
        s, opt, cost = self.sess.run([self.merged_summary, self.optimizer, self.cost],feed_dict={self.x: X})
        self.train_writer.add_summary(s,epoch)
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"], name="z_mu")
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
       # tf.summary.image("x_sample",X[0].reshape(28, 28))
        x_reconstr =  self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X})
       # tf.summary.image("x_reconstr_from_reconstruct",x_reconstr[0].reshape(28, 28))
        return x_reconstr

    def init_session(self, epoch, n_z):
        #print(self.save_path+"_epoch_"+str(epoch)+"_nz_", str(n_z)+'.index')
        self.sess = tf.InteractiveSession()
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        #self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        #self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "my.test.com:6064")
        merged = tf.summary.merge_all()
        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./test_save_path/graph/train",self.sess.graph)
        self.test_writer = tf.summary.FileWriter("./test_save_path/graph/test")


        if not self.restore or self.save_path is None or not os.path.isfile(self.save_path+"_epoch_"+str(epoch)+"_nz_"+ str(n_z)+'.index'):
        # Initializing the tensor flow variables
                print("---- VAE init session ----")
                init = tf.global_variables_initializer()   
                self.sess = tf.InteractiveSession()        
                self.sess.run(init)
        else:
            # Restore variables from disk
            tf.train.Saver().restore(self.sess, self.save_path+"_epoch_"+str(epoch)+"_nz_"+ str(n_z)+"_b_size_"+str(batch_size))
            print("\n---- VAE "+self.save_path+" RESTORED ----\n")



    def save_session(self, epoch,n_z, batch_size):
        # Save the variables to disk.
        #pdb.set_trace()
        if self.save_path is not None:
            try:
                tf.train.Saver().save(self.sess, self.save_path+"_epoch_"+str(epoch)+"_nz_"+ str(n_z)+"_b_size_"+str(batch_size))
            except:
                print("Unexpected error:", sys.exc_info()[0])
            #data_driver.save_data("./training_errors/"+self.save_path+"_epoch_"+str(epoch), "errors", data)

            print("\n---- VAE SAVED IN FILE: "+self.save_path+"_epoch_"+str(epoch)+"_nz_"+ str(n_z) + " ----\n")
            #print("\n---- TRAINING ERRORS SAVED IN FILE: ./training_errors/"+self.save_path+"_epoch_"+str(epoch)+" ----\n")





    def show_data(self, data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices, x_samples, nb_frames = [], mov_types = None, plot_variance=False, nb_samples_per_mov=1, show=True, displayed_movs=None, dynamic_plot=False, plot_3D=False, window_size=10, time_step=5, body_lines=True, only_hard_joints=True, transform_with_all_vtsfe=True, average_reconstruction=True, data_inf=[]):
        """
            Data space representation.
            You can optionally plot reconstructed data as well at the same time.
        """
        if(mov_types == None):
                mov_types = data_driver.mov_types
        if(nb_frames == []):
                nb_frames = x_samples.shape[1]
        #labels = data_driver.data_labels
        
        # don't display more movements than there are
        if nb_samples_per_mov > len(sample_indices):
            display_per_mov = len(sample_indices)
        else:
            display_per_mov = nb_samples_per_mov
        print(str(display_per_mov))
        
        nb_sub_sequences = 1
        
        nb_colors = display_per_mov +1
        if len(reconstr_datasets) > 0:
            nb_colors *= len(reconstr_datasets)
        colors = cm.rainbow(np.linspace(0, 1, nb_colors))
        for j,reco in enumerate(reconstr_datasets):
            # reco shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            print("\n-------- "+reconstr_datasets_names[j])
            print("Dynamics (Sum of variances through time) = "+str(np.sum(np.var(reco, axis=2))))

        if plot_3D:
            print("in plot3D")
            # to avoid modifying x_samples
            data = np.copy(x_samples) #70.70.69
            # x_samples shape = [nb_samples, nb_frames, segment_count, 3 (coordinates)]

            data = data.reshape([len(mov_types)*nb_samples_per_mov, nb_frames, -1, 3]) #70.70.22.3
            
            #TODO: Add variables to defin the shape of data_inf
        
            if(data_inf != []): 
                tmp_shape = data_inf.shape
                data_inf = data_inf.reshape(tmp_shape[0],tmp_shape[1],23,3)
        
                if(tmp_shape[0]== 7):
                    data_inf_tmp = np.zeros([70,70,23,3])
                    for i in np.arange(7):
                        data_inf_tmp[i*10+8] = data_inf[i]
                    data_inf = data_inf_tmp
                    
            data_reconstr = []
            for j,reco in enumerate(reconstr_datasets): #1.1.70.70.66
                data_reconstr.append(reco.reshape([nb_sub_sequences, len(mov_types)*nb_samples_per_mov, nb_frames, -1, 3]))
            segment_count = len(data[0, 0])#22
        
            cs = []
        
            if not body_lines:
                # color every segment point, a color per sample
                for i in range(display_per_mov):
                    for j in range(segment_count):
                        cs.append(colors[i])
        
            # plots = []
            # plot_recs = []
        
            body_lines_indices = [[0, 7], [7, 11], [11, 15], [15, 19], [19, 23]]
            additional_lines = [[15, 0, 19], [7, 11]]
            nb_body_lines = len(body_lines_indices)
            nb_additional_lines = len(additional_lines)
        
            if body_lines:
                def plot_body_lines(plots, j, k, data):
                    
                    #print("mov="+str(mov))

                    for i in range(nb_body_lines):
                        line_length = body_lines_indices[i][1] - body_lines_indices[i][0]
                        # NOTE: there is no .set_data() for 3 dim data...
                        # plot 2D
                        plots[i].set_data(
                            data[data_driver.mov_indices[mov] + j, k, body_lines_indices[i][0] : body_lines_indices[i][1], 0],
                            data[data_driver.mov_indices[mov] + j,k, body_lines_indices[i][0] : body_lines_indices[i][1],1 ]
                        )
                        # plot the 3rd dimension
                        plots[i].set_3d_properties(
                            data[
                                data_driver.mov_indices[mov] + j,
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
                                data_driver.mov_indices[mov] + j,
                                k,
                                additional_lines[i],
                                0
                            ],
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                additional_lines[i],
                                1
                            ]
                        )
                        # plot the 3rd dimension
                        plots[nb_body_lines+i].set_3d_properties(
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                additional_lines[i],
                                2
                            ]
                        )
        
        
                def animate(k):
                    index = 0
                    step = nb_body_lines + nb_additional_lines
                    for j in range(display_per_mov):
                        next_index = index + step
                        plot_body_lines(plots[index : next_index], sample_indices[j], k, data)
                        if(data_inf != []):
                            plot_body_lines(plot_inf[index : next_index], sample_indices[j], k, data_inf)
        
                        for r,reco in enumerate(data_reconstr):
                            for sub in range(nb_sub_sequences):
                                plot_body_lines(plot_recs[r*nb_sub_sequences + sub][index : next_index], r*nb_sub_sequences + sample_indices[j], k, reco[sub])
                        index = next_index
                    title.set_text("Time = {}".format(k))
                    if not show:
                        ax.view_init(30, -150 + 0.7 * k)
            else:
                def animate(k):
                    indices = [data_driver.mov_indices[mov] + j for j in sample_indices]
                    plots[0]._offsets3d = (
                        data[indices, k, :, 0].reshape([segment_count*display_per_mov]),
                        data[indices, k, :, 1].reshape([segment_count*display_per_mov]),
                        data[indices, k, :, 2].reshape([segment_count*display_per_mov])
                    )
                    for r,reco in enumerate(data_reconstr):
                        for sub in range(nb_sub_sequences):
                            plot_recs[r*nb_sub_sequences + sub]._offsets3d = (
                                reco[sub, indices, k, :, 0].reshape([segment_count*display_per_mov]),
                                reco[sub, indices, k, :, 1].reshape([segment_count*display_per_mov]),
                                reco[sub, indices, k, :, 2].reshape([segment_count*display_per_mov])
                            )
                    title.set_text("Time = {}".format(k))
                    if not show:
                        ax.view_init(30, -150 + 0.7 * k)
        
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
                    ax.set_xlim3d(-box_s, box_s)
                    ax.set_ylim3d(-box_s, box_s)
                    ax.set_zlim3d(-box_s, box_s)
                    # set point-of-view: specified by (altitude degrees, azimuth degrees)
                    ax.view_init(30, -150)
                    title = ax.set_title("Time = 0")
                    if body_lines:
                        #Prepare bodylines for the x_original plot
                        for j in range(display_per_mov):
                            # plot the nb_body_lines+nb_additional_lines lines of body segments
                            for k in range(nb_body_lines + nb_additional_lines):
                                # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plots.append(ax.plot([], [], [], c=colors[j], marker='o')[0])
        
                        #Prepare bodylines for the x inf plot
                        for j in range(display_per_mov):
                        # plot the nb_body_lines+nb_additional_lines lines of body segments
                            for k in range(nb_body_lines + nb_additional_lines):
                                # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plot_inf.append(ax.plot([], [], [], c=colors[display_per_mov], marker='D', linestyle='dashed')[0])
        
        
                        # Prepare bodylines for the x reconstr plot
                        for r in range(len(reconstr_datasets)):
                            label = reconstr_datasets_names[r]
                            for sub in range(nb_sub_sequences):
                                plts = []
                                for j in range(display_per_mov):
                                    # plot the nb_body_lines + nb_additional_lines lines of body reconstructed segments
                                    for k in range(nb_body_lines + nb_additional_lines):
                                        if j != 0 or k != 0 or sub != 0:
                                            label = None
                                        plts.append(ax.plot([], [], [], label=label, c=colors[r*nb_sub_sequences + j], marker='D', linestyle='dashed')[0])
                                # plot_recs shape = [nb_reconstr_data*nb_sub_sequences, display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plot_recs.append(plts)
                    else:
                        # just scatter every point
                        plots.append(ax.scatter([], [], [], c=cs))
                        for r in range(len(reconstr_datasets)):
                            label = reconstr_datasets_names[r]
                            plot_recs.append(ax.scatter([], [], [], label=label, c=cs, marker='D'))
        
                    plt.legend()
                    # call the animator.  blit=True means only re-draw the parts that have changed.

                    anim = animation.FuncAnimation(fig, animate, frames=nb_frames, interval=time_step, blit=False)
                    # Save as mp4. This requires mplayer or ffmpeg to be installed
                    if show:
                        plt.show()
                        _ = input("Press [enter] to continue.") # wait for input from the user
                    plt.close()    # close the figure to show the next one.
        
                # turn off interactive mode
                plt.ioff()
            else:
                print("dont display mov")

