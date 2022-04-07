from imp import load_dynamic
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops.gen_math_ops import Sum
import os
import wandb    
import datetime
import numpy as np
import pprint
import matplotlib.pyplot as plt

from numpy import loadtxt
from tensorflow.keras.models import load_model
from icecream import ic
import streamlit as st

class Train_PINN():
    def __init__(self, r0, rf, dt, fun_r, phi_r_weighting, mu, config, short_time, short_way):

        self.fun_r = fun_r
        self.phi_r_weighting  = phi_r_weighting
        self.mu = mu
        self.r0 = r0
        self.rf = rf
        self.ModelInput = tf.reshape(tf.convert_to_tensor(tf.concat([r0, rf], axis=0)), (1,-1))
        self.dt = dt
        self.short_time = short_time
        self.short_way = short_way


        self.config = config
        self.optimizer   = config['optimizer']
        self.lr          = config['learning rate']
        self.Layer_Units = config['Layer_Units']
        self.epochs      = int(config['epochs'])

        if not short_time:
            self.epochs      = 3*self.epochs
            self.lr          = 1*self.lr

        #TotalParams = int(np.floor(CountParams(x_train.shape[1], self.Layer_Units, y_train.shape[1])/1000))

        #execution_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.Param_name = str(execution_time) + "-" + str(self.optimizer) + "-" +str(TotalParams) + "k-Params-" + str(self.epochs) + "-Epochs-" + str(self.batch_size)


    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Build(self,inputs,outputs):
        # Initialize a feedforward neural network
        self.Dimensions=outputs
        
        # Introduce a scaling layer to map input to [lb, ub]
        # scaling_layer = tf.keras.layers.Lambda(
        #             lambda x: 2*(x - lb)/(ub - lb) -1.0)

        # x1 = scaling_layer(inputs)
        x1 = inputs

        # Append hidden layers
        for _layer in range(0,len(self.Layer_Units)):
            # x1 = tf.keras.layers.BatchNormalization()(x1)
            x1 = tf.keras.layers.Dense(units=self.Layer_Units[_layer],
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(x1)

        # Output is three-dimensional
        outputs = tf.keras.layers.Dense(units=outputs, name='SMA', activation='sigmoid')(x1)

        model = tf.keras.Model(inputs=inputs, outputs=outputs )
        
        return model

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    
    def get_r(self):
        output = self.model(self.ModelInput)
        # st.write(output)
        sin_alpha2, sin_beta2 = output[0,0], output[0,1] # multipy beta by 1 or -1
        return self.fun_r(self.r0, self.rf, self.mu, self.dt, sin_alpha2, sin_beta2, 
                          short_time = self.short_time,
                          short_way = self.short_way)

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def compute_loss(self):
        # Compute phi^r
        r = self.get_r()

        phi_r = tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        loss = phi_r*self.phi_r_weighting
        return loss

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def get_grad(self):
    
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.compute_loss()

        g = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, g

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def train(self, model, print_interval):
        st.markdown('# Training Process')
        
        self.c = st.empty()
        self.c2 = st.empty()
        self.model = model                
        # Choose the optimizer  
        optim = getattr(tf.keras.optimizers,self.optimizer)(learning_rate=self.lr)


        # Define one training step as a TensorFlow function to increase speed of training
        #@tf.function
        def train_step():
            # Compute current loss and gradient w.r.t. parameters
            loss, grad_theta = self.get_grad()
            return loss, grad_theta

        # Number of training epochs
        N = self.epochs
        Loss = np.zeros((N,1))

        for i in range(N):
            
            loss, grad_theta = train_step()   

            # Append current loss to hist
            Loss[i] = loss

            if loss<1e-25:
                st.write(f'Stopped Early at It ',i, ': loss = ',loss.numpy())

                output = self.model(self.ModelInput)
                sin_alpha2, sin_beta2 = output[0,0].numpy(), output[0,1].numpy()
                alpha = 2*np.arcsin(sin_alpha2)
                beta = 2*np.arcsin(sin_beta2)

                st.write(fr'$\alpha$ = ',alpha*180/np.pi,r', $\beta$  = ',beta*180/np.pi)
                st.markdown('#')
                break
            
            # Output current loss after x iterates
            if i%print_interval == 0:
                self.c.write(f'It {i:05d}: loss = {loss:10.8e}')

                output = self.model(self.ModelInput)
                sin_alpha2, sin_beta2 = output[0,0].numpy(), output[0,1].numpy()
                alpha = 2*np.arcsin(sin_alpha2)
                beta = 2*np.arcsin(sin_beta2)
                self.c2.write(fr'$\alpha$ = {alpha*180/np.pi:.3f}, $\beta$  = {beta*180/np.pi:.3f}')

            # Perform gradient descent step
            #st.write(grad_theta)

            # Apply gradients for next training loop
            
            optim.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            if tf.math.is_nan(loss):
                break

        return Loss, i
                
class PINN_Eq():
    def __init__(self, model, lb, ub):
        self.lb = lb
        self.ub = ub
        self.model = model
    def Sample(self, N, DTYPE):
        self.N  = N
        self.tspace = np.linspace(self.lb, self.ub, int(self.N) + 1).reshape(-1,1)
        xspace = np.linspace(-3, 3, int(self.N) + 1)   
        yspace = np.linspace(-3, 3, int(self.N) + 1)   
        zspace = np.linspace(-3, 3, int(self.N) + 1)   
        X, Y,Z = np.meshgrid(xspace, yspace, zspace)
        Xgrid = np.vstack([X.flatten(),Y.flatten(), Z.flatten()]).T

        # Determine predictions of u(t, x)
        upred = self.model.predict(tf.cast(self.tspace,DTYPE))

        #Reshape upred
        self.U = upred

    def Plot(self, orientation):
        # Surface plot of solution u(t,x)
        fig = plt.figure(figsize=(9,6))
        # ax = fig.add_subplot(121, projection='3d')
        ax = fig.add_subplot(111)
        # ax.scatter(self.U[:,0], self.U[:,1], self.U[:,2])
        # ax.scatter(self.U[:,0], self.U[:,1], self.U[:,2])
        ax.plot(self.tspace, self.U)

        #ax.view_init(orientation[0],orientation[1])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        #ax.set_zlabel('$z_\\theta(t,x)$')
        ax.set_title('Solution of Burgers equation')
        #plt.savefig('Burgers_Solution.pdf', bbox_inches='tight', dpi=300);
        #ax2 = fig.add_subplot(122)
        #ax2.set_title('Solution of Burgers equation')
        plt.show(block=False)
        plt.pause(1)


            

