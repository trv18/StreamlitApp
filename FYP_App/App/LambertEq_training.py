#!/usr/bin/env python
# coding: utf-8

# In[2]:

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"calling", category=FutureWarning)

# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
import pykep as pk
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
from LambertEq_tools import Train_PINN
import random
from numpy.linalg import norm as norm
import streamlit as st
from datetime import datetime




# Set data type
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)


# Define residual of the PDE
def fun_r(r0, rf, mu, dt, sin_alpha2, sin_beta2, short_way=True, short_time=True):

    c = tf.norm(rf-r0)
    s = (tf.norm(r0) + tf.norm(rf) + c)/2

    alpha = 2*tf.asin(sin_alpha2)
    beta  = 2*tf.asin(sin_beta2) 

    if (    short_way and not short_time)  : alpha = 2*pi - alpha
    if (not short_way and     short_time)  : beta  = -1*beta
    if (not short_way and not short_time)  : alpha, beta = 2*pi - alpha, -1*beta

    a = s / (2*tf.sin(alpha/2)**2)
    a2 = (s-c) / (2*tf.sin(beta/2)**2)

    sin_alpha = tf.sin(alpha)
    sin_beta = tf.sin(beta)
    
    res = tf.abs(tf.math.sqrt(mu)*dt - a**1.5 * (alpha - beta - (sin_alpha - sin_beta)))
    a_check = tf.abs(a-a2)

    return a_check + res


# In[3]:
@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def plot3D_grav(u0, v0, ub, uf=[None,None,None], mode='SUN', ax=None): 
    plt.style.use('dark_background')
    x0, y0, z0 = u0
    vx0, vy0, vz0 = v0


    # mu = pk.MU_SUN/pk.AU**3 * deltat**2
    if mode=='SUN':
        def f(state, t):
            # x, dx, y, dy = state  # Unpack the state vector
            x, dx, y, dy, z, dz = state  # Unpack the state vector
            rx = x
            ry = y
            rz = z
            return dx, -pk.MU_SUN*rx/(rx**2 + ry**2 + rz**2)**(3/2),                    dy, -pk.MU_SUN*ry/(rx**2 + ry**2 + rz**2)**(3/2),                    dz, -pk.MU_SUN*rz/(rx**2 + ry**2 + rz**2)**(3/2) # Derivatives
            
    elif mode=='EARTH':
        def f(state, t):
            # x, dx, y, dy = state  # Unpack the state vector
            x, dx, y, dy = state  # Unpack the state vector
            rx = x + pk.EARTH_RADIUS
            ry = y + pk.EARTH_RADIUS
            return dx, -pk.MU_EARTH*rx/(rx**2 + ry**2)**(3/2), dy, -pk.MU_EARTH*ry/(rx**2 + ry**2)**(3/2) # Derivatives

    #state0 = [u0, 0]
    state0 = [x0, vx0, y0, vy0, z0, vz0]
    t = np.arange(0.0, ub+ub/50000, ub/50000)

    states = odeint(f, state0, t)

    if ax==None:
        # ic()
        fig = plt.figure(figsize=(10,5))
        ax = plt.gca()

    ax.ticklabel_format(useOffset=False)
    ax.plot(states[:, 0], states[:, 2], color='green', label = 'Transfer', linewidth=2)
    ax.plot(states[0, 0], states[0, 2], color='blue', label = 'Initial Position', marker='o', markersize=10)
    # ax.plot(states[-1, 0], states[-1, 2], color='red', label = 'Final Position', marker='o', markersize=10)
    if all(uf):
        ax.plot([0, states[0,0]], [0, states[0,2]], '-r')
        ax.plot([0, uf[0]], [0, uf[1]], '-r')
        ax.plot(uf[0], uf[1], color='orange', label = 'Lambert Position', marker='o', markersize=10)

    ax.legend(fontsize = 15)
    ax.xaxis.offsetText.set_fontsize(20)
    ax.yaxis.offsetText.set_fontsize(20)

    # ic(states[-1,:])
    # ax.view_init(0,0)

    return states, fig


# ## 2. Generate a Lambert Problem
# 
# We use Izzo's pykep alogirthm to generate boundary condtions for the PINNS to solve
# 

# In[4]:


class LambertEq():
    def __init__(self):
        random.seed()
        np.random.seed()

# In[5]:


class LambertEq(LambertEq):
    def Get_Lambert(self, new=True, print=True, clockwise=True, shortway=True, inputs={}):
        
        self.inputs = inputs
        ############################################################################
        ########################## Define Lambert Problem ##########################
        ############################################################################

        # Set limits on TOF and starting date
        if inputs['DateType']=='Specified':

            start = str(inputs["StartEpoch"]) + ' 00:00:00'
            end = str(inputs["EndEpoch"]) + ' 00:00:00'

            self.t1     = int(pk.epoch_from_string(start).mjd)
            self.t2     = int(pk.epoch_from_string(end).mjd)
            self._TOF   = self.t2-self.t1

        else:
            Start_Epoch     = int(pk.epoch_from_string("2001-01-01 00:00:00").mjd)
            End_Epoch       = int(pk.epoch_from_string("2031-01-01 00:00:00").mjd)
            TOF_range       = [10, 300]
        
        # specify target planet
        _target = 'mars'

        # only generate new problem if required
        if inputs['DateType']=='Random' and new:
            self.t1              = np.random.randint(low=Start_Epoch, high=End_Epoch, size=1)
            self._TOF            = np.random.randint(low=TOF_range[0], high=TOF_range[1])
            self.t2              = self.t1 + self._TOF

        # Get Ephemeris data from pykep
        Departure       = pk.planet.jpl_lp('earth') 
        Target          = pk.planet.jpl_lp(_target)

        States0         = Departure.eph(pk.epoch(int(self.t1), 'mjd'))
        Statesf         = Target.eph(pk.epoch(int(self.t2), 'mjd'))

        self._r0              = np.array(States0[0])
        self._rf              = np.array(Statesf[0])

        ################################################################################
        ############################# Solve Lambert Problem ############################
        ################################################################################

        lamsol_list = pk.lambert_problem(r1=self._r0, r2=self._rf, tof=int(self._TOF)*24*3600,
                                        mu=pk.MU_SUN, cw=clockwise)
        self.v1 = lamsol_list.get_v1()[0]
        v2 = lamsol_list.get_v2()[0]
                            
        self.TOF = self.t2-self.t1
        ################################################################################
        ############################# Set up training data #############################
        ################################################################################


        # Define normamlising constants for distance and time
        self.nc = np.linalg.norm(self._r0)
        self.tnc = np.sqrt(self.nc**3/pk.MU_SUN)

        # Normalise input data
        self.r0 = self._r0/self.nc
        self.rf = self._rf/self.nc   
        self.mu = pk.MU_SUN/(self.nc)**3 * self.tnc**2

        # Specifiy short or long way solution (dtheta<180?)
        self.short_way=shortway

        ################################################################################
        ############################# Ensure solution exists ###########################
        ################################################################################


        c = norm(self._r0 - self._rf)
        s = (norm(self._r0) +  norm(self._rf) + c) /2

        alpha_m = np.pi
        beta_m = 2*np.arcsin(np.sqrt((s-c)/s))

        # Minimum Energy Solution - determines long or short time solution 
        dt_m = np.sqrt(s**3/(8*pk.MU_SUN))*(np.pi - beta_m + np.sin(beta_m))
                
        dtheta = np.arccos(np.dot(self._r0,self._rf)/(norm(self._r0)*norm(self._rf)))

        # if long way specified, adjust change in true anomaly for parabolic transfer time calculation
        if not self.short_way:
            dtheta = 2*np.pi - dtheta
        # parabolic transfer time - minimum 
        dt_p = np.sqrt(2)/3*(s**1.5 - np.sign(np.sin(dtheta))*(s-c)**1.5)/np.sqrt(pk.MU_SUN)

        # Determine if desired solution corresponds to short or long time solution
        if self.TOF < dt_m/3600/24 :
            self.short_time = True
        else:
            self.short_time = False
            
        # If desired solution non existent generate new solution with identical parameters
        if self.TOF < dt_p/3600/24:
            if inputs["DateType"] == "Specified":
                st.markdown(f'<p style="text-align:left;color:#ff000d ;font-size:15px;border-radius:0%;"> \
                        Please Enter Another Set of Dates. Time of Flight of {self.TOF} days is lower than Parabolic time of {dt_p/3600/24:.2f} days\
                         </p>', unsafe_allow_html=True)
            else:
                self.Get_Lambert(shortway = self.short_way, inputs=self.inputs)

            # Avoids double execution of function
            return 0
        ############################################################################
        ############################# Outputting ###################################
        ############################################################################

        if print:
            
            st.markdown('# Problem Definition')


            st.markdown(f'<p style="text-align:left;color:#00cf22 ;font-size:20px;border-radius:0%;"> \
                        Start Date: {pk.epoch(int(self.t1), "mjd")} <br> \
                        End Date: &nbsp; {pk.epoch(int(self.t2), "mjd")} \
                         </p>', unsafe_allow_html=True)

            st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                        Min E    &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp     &Delta;T: {dt_m/3600/24:.3f} days <br> \
                        Parabolic &Delta;T: {dt_p/3600/24:.3f} days <br> \
                        Desired    &nbsp&nbsp 	&Delta;T:  {int(self.TOF):.3f} days <br> \
                         </p>', unsafe_allow_html=True)

            if not self.short_way:
                st.markdown('<p style="text-align:left;color:#ff000d;font-size:20px;border-radius:0%;"> \
                            Adjusting solution process for long way solution<p>', unsafe_allow_html=True)

            st.markdown("#")
            # latext = r'''
            # ## Latex example
            # ### full equation 
            # $$ 
            # \Delta G = \Delta\sigma \frac{a}{b} 
            # $$ 
            # ### inline
            # Assume $\frac{a}{b}=1$ and $\sigma=0$...  
            # '''



        _, fig = plot3D_grav(self._r0, self.v1, self.TOF*24*3600, self._rf)
        # avoid fading when reloading - see link https://tinyurl.com/3a3u3waf
        st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)
        st.pyplot(fig)



# ### 3. Set up network architecture

# In[6]:


class LambertEq(LambertEq):
    def TrainModel(self, num_epochs=1001, lr = 2e-2):

        ############################################################################
        ############################# Set Seed for reproducibility #################
        ############################################################################

        n = 415
        # st.write(f'seed = {n}')
        random.seed(n)
        np.random.seed(n)
        tf.random.set_seed(n)

        ############################################################################
        ############################# Set up Model config options ##################
        ############################################################################
        config = {}
        config['optimizer']    = 'Adam'
        # config['learning rate']=  tf.keras.optimizers.schedules.PiecewiseConstantDecay([500.0, 1000.0, 1500.0],[1e-1,1e-2, 5e-3, 1e-4])
        config['learning rate']= lr
        config['Layer_Units']  = np.repeat([50], 1)
        config['epochs']       = num_epochs

        # normalise TOF
        self.dt = self.TOF*24*3600/self.tnc

        ############################################################################
        ################################# Train Model ##############################
        ############################################################################
        
        # Create instance of class to train PINN
        self.PINN = Train_PINN(self.r0, self.rf, self.dt,
                        fun_r = fun_r,
                        phi_r_weighting = 1,
                        mu = self.mu,
                        config=config,
                        short_time = self.short_time, 
                        short_way = self.short_way)

        # Compile model
        model = self.PINN.Build(inputs=tf.keras.Input(6), outputs=2) 
        
        # Train model
        Loss, epochs_trained = self.PINN.train(print_interval=200, model=model)
        
        return Loss, epochs_trained


# In[7]:


class LambertEq(LambertEq):
    def Get_Error(self, checked_long_way=False):

        ############################################################################
        ############################# Get Correct SMA ##############################
        ############################################################################

        c = tf.norm(self._rf-self._r0)
        s = (norm(self._r0) + norm(self._rf) + c)/2
        avec = pk.ic2par(self._r0, self.v1, mu = pk.MU_SUN)[0]

        alpha_true = 2*np.arcsin(np.sqrt(s/(2*norm(avec))))
        beta_true = 2*np.arcsin(np.sqrt((s-c)/(2*norm(avec))))

        if (    self.short_way and not self.short_time)  : alpha_true = 2*pi - alpha_true
        if (not self.short_way and     self.short_time)  : beta_true  = -1*beta_true
        if (not self.short_way and not self.short_time)  : alpha_true, beta_true = 2*pi - alpha_true, -1*beta_true
        ############################################################################
        ########################### Get Predicted SMA ##############################
        ############################################################################

        output = self.PINN.model(tf.reshape(tf.convert_to_tensor(tf.concat([self.r0, self.rf], axis=0)), (1,-1)))
        sin_alpha2, sin_beta2 = output[0,0].numpy(), output[0,1].numpy()

        alpha = 2*np.arcsin(sin_alpha2)
        beta = 2*np.arcsin(sin_beta2)

        # Adjust for long way or long time 
        if (    self.short_way and not self.short_time)  : alpha = 2*pi - alpha
        if (not self.short_way and     self.short_time)  : beta  = -1*beta
        if (not self.short_way and not self.short_time)  : alpha, beta = 2*pi - alpha, -1*beta

        c = tf.norm(self.rf-self.r0)
        s = (norm(self.r0) + norm(self.rf) + c)/2

        a = self.nc* s / (2*np.sin(alpha/2)**2)
        a2 = self.nc* (s-c) / (2*np.sin(beta/2)**2)

        residual = np.sqrt(pk.MU_SUN)*self.dt*self.tnc - (a)**1.5 * (alpha - beta - (np.sin(alpha) - np.sin(beta)))   

        ############################################################################
        ############################# Output Performance ###########################
        ############################################################################

        # only print out once
        if not checked_long_way:
             
            st.markdown('# Results and Performance')
            st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                        &alpha; = {alpha*180/pi:5f}, &beta; = {beta*180/pi:5f} <br> \
                        &alpha; true = {alpha_true*180/pi:5f}, &beta; true = {beta_true*180/pi:5f}\
                         </p>', unsafe_allow_html=True)
            # st.markdown('#')

            st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                        sin(&alpha;)      = {np.sin(alpha):5f}, sin(&beta;) = {np.sin(beta):5f} <br> \
                        sin(&alpha;) true = {np.sin(alpha_true):5f}, sin(&beta;) true= {np.sin(beta_true):5f} \
                         </p>', unsafe_allow_html=True)

            # st.markdown('#')
            st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                        SMA         = {a/1e11} <br> \
                        SMA check   = {a2/1e11} <br> \
                        correct SMA = {norm(avec)/1e11} \
                         </p>', unsafe_allow_html=True)
            # st.markdown('#')

            st.write(f'res = {float(residual):.5E}  \n')
            st.markdown('#')

        ############################################################################
        ######################### Get Velocity prediction ##########################
        ############################################################################

        # calculate normalised vectors for velocity calculation
        u1 = self._r0/np.linalg.norm(self._r0)
        uc = (self._rf-self._r0)/np.linalg.norm(self._rf - self._r0)

        A = np.sqrt(pk.MU_SUN/(4*a))/np.tan(alpha/2)
        B = np.sqrt(pk.MU_SUN/(4*a))/np.tan(beta/2)

        v1_nn = (B+A)*uc + (B-A)*u1
        error = norm((self.v1-v1_nn)/self.v1)*100/3

        # if error low enough solution is correct
        if error < 0.1:
            st.markdown(f'<p style="text-align:left;color:#00cf22 ;font-size:20px;b <br>order-radius:0%;"> \
                        V0 desired &nbsp;&nbsp;&nbsp;  :{np.round(self.v1,3)} km/s <br> \
                        V0 predicted :{np.round(v1_nn,3)} km/s <br> \
                        error = {error:.10f}%  \
                        </p>', unsafe_allow_html=True)

        # if error too high check that solution does not correspond to counterclockwise solution 
        elif checked_long_way == False:
            st.write(f'error = {error:.10f}%')
            st.write('Checking error for complementary solution  \n')

            # Ensure no new problem is generated
            self.Get_Lambert(new=False, print=False, clockwise=False, shortway = self.short_way, inputs=self.inputs)

            # recalculate error with new solution
            self.Get_Error(checked_long_way=True)

            # do not not execute rest of loop
            return 0

        # if error still too high problem has not converged
        else:
            st.markdown(f'<p style="text-align:left;color:#ff000d ;font-size:20px;b <br>order-radius:0%;"> \
                        V0 desired   :{np.round(self.v1,3)} <br> \
                        V0 predicted :{np.round(v1_nn,3)} <br> \
                        error = {error:.10f}%  \
                        </p>', unsafe_allow_html=True)