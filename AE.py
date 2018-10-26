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

class AE(object):
        """ Autoencoder (AE) 
        """
        def __init__(self,num_input,num_hidden_1,n_z,learning_rate, typeInit = 'randomNormal',typeActivation='sigmoid',typeOpti='RMS',restore=False, epoch=30000):
                self.num_input = num_input
                self.num_hidden_1 = num_hidden_1
                self.n_z = n_z
                self.learning_rate = learning_rate
                self.typeInit = typeInit
                self.typeActivation = typeActivation
                # Construct model
                self.createNN(num_input,num_hidden_1,n_z)
                self.optimizerComputation(typeOpti)
                print(restore)
               # init session to restore or randomly initialize all variables in network
                self.init_session(epoch=epoch, n_z=self.n_z,restore=restore)
 
                
        def optimizerComputation(self, typeOpti='RMS'):
                with tf.name_scope('OptimizerComputation'):
                        if(typeOpti=='RMS'):
                                with tf.name_scope('loss'):
                                        self.loss = tf.reduce_mean(tf.pow(self.X - self.decoder_op, 2))
                                with tf.name_scope('RMS_opti'):
                                        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)    
                        elif(typeOpti=='losses.mean_squared_error'):
                                with tf.name_scope('loss'):
                                        self.loss = tf.losses.mean_squared_error(self.X, self.decoder_op)
                                with tf.name_scope('Adam_opti'):
                                        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                try:
                        tf.summary.histogram("loss",self.loss)
                except:
                        print("Error loss summaries")


        
        def createNN(self,num_input,num_hidden_1,n_z):
                with tf.name_scope('x'):
                        self.X = tf.placeholder(tf.float32, [None, num_input])

                if(self.typeInit=='randomNormal'):
                        with tf.name_scope('weights'):
                                self.weights = {  
                                    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1]),name="encoder_h1"),
                                    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, n_z]),name="encoder_h2"),
                                    'decoder_h1': tf.Variable(tf.random_normal([n_z, num_hidden_1]),name="decoder_h1"),
                                    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input]),name="decoder_h2"),
                                }

                if(self.typeInit=='xavier_init'): 
                        with tf.name_scope('weights'):
                                self.weights = {  
                                    'encoder_h1': tf.Variable(xavier_init(num_input, num_hidden_1),name="encoder_h1"),
                                    'encoder_h2': tf.Variable(xavier_init(num_hidden_1, n_z),name="encoder_h2"),
                                    'decoder_h1': tf.Variable(xavier_init(n_z, num_hidden_1),name="decoder_h1"),
                                    'decoder_h2': tf.Variable(xavier_init(num_hidden_1, num_input),name="decoder_h2"),
                                }

                with  tf.name_scope('biases'):
                        self.biases = {
                            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1]),name="encoder_b1"),
                            'encoder_b2': tf.Variable(tf.random_normal([n_z]),name="encoder_b2"),
                            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1]),name="decoder_b1"),
                            'decoder_b2': tf.Variable(tf.random_normal([num_input]),name="decoder_b2"),
                        }
                     
                try:
                        tf.summary.histogram("w_enc_h1",self.weights['encoder_h1'])
                        tf.summary.histogram("w_enc_h2",self.weights['encoder_h2'])
                        tf.summary.histogram("w_enc_b1",self.weights['encoder_b1'])
                        tf.summary.histogram("w_enc_b2",self.weights['encoder_b2'])
                        tf.summary.histogram("w_dec_h1",self.weights['decoder_h1'])
                        tf.summary.histogram("w_dec_h2",self.weights['decoder_h2'])
                        tf.summary.histogram("w_dec_b1",self.weights['decoder_b1'])
                        tf.summary.histogram("w_dec_b2",self.weights['decoder_b2'])
                except:
                        print("Error  w enc summaries")
        
                self.encoder_op = self.encoder(self.X)
                self.decoder_op = self.decoder(self.encoder_op)
                

        # Building the encoder
        def encoder(self,x, nbLayer=2):
                layer = []
                with tf.name_scope('Encoder'):
                        if(self.typeActivation=="sigmoid"):
                                layer_tmp = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h'+ str(1)]), self.biases['encoder_b'+ str(1)]))
                                layer.append(layer_tmp)
        
                                for i in range(1, nbLayer):
                                        layer_tmp = tf.nn.sigmoid(tf.add(tf.matmul(layer[i-1], self.weights['encoder_h'+ str(i+1)]), self.biases['encoder_b'+ str(i+1)]))
                                        layer.append(layer_tmp)
        
                        elif(self.typeActivation=="relu"):
                                layer_tmp = tf.nn.relu_layer(x, self.weights['encoder_h'+ str(1)], self.biases['encoder_b'+ str(1)], name='l'+ str(1)+'_enc_relu')
                                layer.append(layer_tmp)
        
                                for i in range(1,nbLayer):
                                        layer_tmp = tf.nn.relu_layer(layer[i-1], self.weights['encoder_h'+ str(i+1)], self.biases['encoder_b'+ str(i+1)], name='l'+ str(i+1)+'_enc_relu')
                                        layer.append(layer_tmp)
        
        
                        elif(self.typeActivation=="leaky"):
                                val = tf.add(tf.matmul(x, self.weights['encoder_h'+ str(1)]) ,self.biases['encoder_b'+ str(1)])
                                layer_tmp = tf.nn.leaky_relu(val,alpha=0.5, name='l'+ str(1)+'_enc_leaky')
                                layer.append(layer_tmp)
        
                                for i in range(1,nbLayer):
                                        val = tf.add(tf.matmul(layer[i-1], self.weights['encoder_h'+ str(i+1)]),self.biases['encoder_b'+ str(i+1)])
                                        layer_tmp = tf.nn.leaky_relu(val,alpha=0.5, name='l'+ str(i+1)+'_enc_leaky')
                                        layer.append(layer_tmp)
                        for i in range(nbLayer):
                                with tf.name_scope('enclayer_'+str(i+1)):
                                        layer[i]
                with tf.name_scope('z'):
                        self.z = layer[nbLayer-1]

                imgTest = tf.reshape(x, [-1,23,3,1])
                try:
                        tf.summary.image("x", imgTest)
                except:
                        print("Error x summary")

                return self.z
        
        
        # Building the decoder
        def decoder(self,x, nbLayer=2, typeActivation="sigmoid"):
                with tf.name_scope('Decoder'):
                        layer = []
                        if(self.typeActivation=="sigmoid"):
                                layer_tmp = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h'+ str(1)]), self.biases['decoder_b'+ str(1)]))
                                layer.append(layer_tmp)
                                for i in range(1, nbLayer):
                                        layer_tmp = tf.nn.sigmoid(tf.add(tf.matmul(layer[i-1], self.weights['decoder_h'+ str(i+1)]),  self.biases['decoder_b'+ str(i+1)]))
                                        layer.append(layer_tmp)                                
                        elif(self.typeActivation=="relu"):
                                layer_tmp = tf.nn.relu_layer(x,  self.weights['decoder_h'+ str(1)], self.biases['decoder_b'+ str(1)], name='l'+ str(1)+'_dec_relu')
                                layer.append(layer_tmp)                                
                                for i in range(1, nbLayer):
                                        layer_tmp = tf.nn.relu_layer(layer[i-1],  self.weights['decoder_h'+ str(i+1)], self.biases['decoder_b'+ str(i+1)], name='l'+ str(i+1)+'_dec_relu')
                                        layer.append(layer_tmp)    
                        elif(self.typeActivation=="leaky"):
                                val = tf.add(tf.matmul(x, self.weights['decoder_h'+ str(1)]) ,self.biases['decoder_b'+ str(1)])
                                layer_tmp = tf.nn.leaky_relu(val,alpha=0.5, name='l'+ str(1)+'_dec_leaky')
                                layer.append(layer_tmp)
        
                                for i in range(1,nbLayer):
                                        val = tf.add(tf.matmul(layer[i-1], self.weights['decoder_h'+ str(i+1)]),self.biases['decoder_b'+ str(i+1)])
                                        layer_tmp = tf.nn.leaky_relu(val,alpha=0.5, name='l'+ str(i+1)+'_dec_leaky')
                                        layer.append(layer_tmp)
                            
                        for i in range(nbLayer):
                                with tf.name_scope('declayer_'+str(i+1)):
                                        layer[i]    

                imgTest = tf.reshape(layer[nbLayer-1], [-1,23,3,1])
                try:
                        tf.summary.image("x_reco", imgTest)
                except:
                         print("Error xreco summary")

        
                return layer[nbLayer-1]
        

   
        
        def partial_fit(self, X, epoch):
                """Train model based on mini-batch of input data.
                
                Return cost of mini-batch.
                """
                
                #opt, cost = self.sess.run((self.optimizer, self.cost),  feed_dict={self.x: X})
                s, opt, loss = self.sess.run([self.merged_summary, self.optimizer, self.loss],feed_dict={self.X: X})
                # if(epoch%5==0):
                self.train_writer.add_summary(s,epoch)
                return loss
        
        def retrieve_latent_space(self,X):
                z = self.sess.run(self.z,feed_dict={self.X: X})
                return z



        def init_session(self, epoch, n_z, restore=False,save_path="./test_ae_save_path/"):

                self.sess = tf.InteractiveSession()
                merged = tf.summary.merge_all()
                self.merged_summary = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter("./test_ae_save_path/graph/train",self.sess.graph)
                self.test_writer = tf.summary.FileWriter("./test_ae_save_path/graph/test")
                all_path = save_path+"ae_epoch_"+str(epoch)+"_nz_"+ str(n_z) + "_" + self.typeInit +"_" + self.typeActivation

                if not restore or save_path is None or not os.path.isfile(all_path+'.index'):
                # Initializing the tensor flow variables
                        print("---- AE init session ----")
                        init = tf.global_variables_initializer()   
                        self.sess = tf.InteractiveSession()        
                        self.sess.run(init)
                else:
                    # Restore variables from disk
                    tf.train.Saver().restore(self.sess, all_path)
                    print("\n---- AE "+all_path+" RESTORED ----\n")
        
        
        def save_session(self,epoch,n_z,  save_path="./test_ae_save_path/"):
            # Save the variables to disk.

            if save_path is not None:
                nameF = save_path+"ae_epoch_"+str(epoch)+"_nz_"+ str(n_z)+ "_" + self.typeInit +"_" + self.typeActivation
                try:
                    tf.train.Saver().save(self.sess, nameF)
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                #data_driver.save_data("./training_errors/"+self.save_path+"_epoch_"+str(epoch), "errors", data)
        
                print("\n---- AE SAVED IN FILE: "+ nameF + " ----\n")
 
