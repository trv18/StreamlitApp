import sys

import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt
import pykep             as pk
import time
import pandas            as pd
import streamlit         as st
import random
import utils
import time 
import queue
from tqdm import tqdm


from tfc              import utfc
from tfc.utils        import TFCDict, egrad, MakePlot, NllsClass, NLLS, LS
from jax              import jit, jacfwd
from jax.numpy.linalg import norm

from threading import Thread


from icecream         import ic
from scipy.integrate  import odeint
from IPython.display  import display 
# from Visualise_Tools  import set_size, format_axes
# from OrbMech_Utilities import plot3D_grav, Get_PropagationError

from astropy import units as u

from poliastro.bodies              import Earth, Mars, Sun

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

Visualise_Tools = utils.import_from_file("Visualise_Tools", "./app/TFC/Visualise_Tools.py")
OrbMech_Utilities = utils.import_from_file("OrbMech_Utilities", "./app/TFC/OrbMech_Utilities.py")

## Import functions - Need to find more efficient way to import
def set_size(width, fraction=1, subplots=(1, 1)):
    Visualise_Tools.set_size(width, fraction, subplots)

def format_axes(ax, fontsize, xlabel, ylabel, scale_legend=False, force_ticks=None):
    Visualise_Tools.format_axes(ax, fontsize, xlabel, ylabel, scale_legend, force_ticks)

def Get_PropagationError():
    OrbMech_Utilities.Get_PropagationError()

def plot3D_grav(u0, v0, ub, uf=[None,None,None], mode='SUN', ax=None, J2=0, R=0, return_t_array=False, Include_J2 = False): 

    if return_t_array:
        states, t, fig = OrbMech_Utilities.plot3D_grav(u0, v0, ub, uf, mode, ax, J2, R, return_t_array, Include_J2)
        return states, t, fig
    else:
        states, fig = OrbMech_Utilities.plot3D_grav(u0, v0, ub, uf, mode, ax, J2, R, return_t_array, Include_J2)
        return states, fig

class LambertEq():
    def __init__(self):
        # Reset workspace 
        onp.random.seed()
        random.seed()
        # tf.compat.v1.reset_default_graph()


def TrainModel(l, points=100, poly_order=30, poly_removed=2, basis_func='CP', method="pinv", plot=True, save_orbit=False, run_type='TFC', inputs={}, container=None):
    container.empty()
    with container.container():
        ########### Define Constants ################
        mu = l.mu
        J2 = Sun.J2.value
        Include_J2 = inputs['Include_J2']
        num_runs = inputs['num_runs']
        
        R_sun = (6963408*1e3) /l.nc # sun radius in m

        r0 = l.r0
        rf = l.rf
        v0 = onp.array(l._v0) # original velocity
        v1 = l.v1 # lambert velocity

        _deltat = l.TOF*24*3600
        ub = float(_deltat/l.tnc)

        # start = time.time()
        # Create the univariate TFC class
        N = points # Number of points in the domain
        m = poly_order # Degree of basis function expansion
        nC = poly_removed # Indicates which basis functions need to be removed from the expansion

        start = time.time_ns()/(10 ** 9)

        myTfc = utfc(N,nC,m, basis=basis_func, x0=0,xf=ub)
        x = myTfc.x # Collocation points from the TFC class

        # Get the basis functions from the TFC class
        H = myTfc.H
        H0 = lambda x: H(np.zeros_like(x))
        Hf = lambda x: H(ub*np.ones_like(x))

        #------------------------------------------------------------------------
        #------------------------------------------------------------------------
        #--------------------- Create the constrained expression ----------------
        #------------------------------------------------------------------------
        #------------------------------------------------------------------------

        g = lambda x,xi: np.array([np.dot(H(x),xi[:,0])])

        rx = lambda x,xi: np.dot(H(x),xi['x'])  \
                    + (ub-x)/ub*( r0[0] - np.dot(H0(x),xi['x']) ) \
                    + (x)/ub*( rf[0] - np.dot(Hf(x),xi['x']) ) \

        ry = lambda x,xi: np.dot(H(x),xi['y'])  \
                    + (ub-x)/ub*( r0[1] - np.dot(H0(x),xi['y']) ) \
                    + (x)/ub*( rf[1] - np.dot(Hf(x),xi['y']) ) \

        rz = lambda x,xi: np.dot(H(x),xi['z'])  \
                    + (ub-x)/ub*( r0[2] - np.dot(H0(x),xi['z']) ) \
                    + (x)/ub*( rf[2] - np.dot(Hf(x),xi['z']) ) \

        # Create the residual
        drx = egrad(rx)
        d2rx = egrad(drx)

        dry = egrad(ry)
        d2ry = egrad(dry)

        drz = egrad(rz)
        d2rz = egrad(drz)
        t0 = onp.zeros_like(x)

        v1_guess = lambda x, xi: np.array([drx(t0, xi), dry(t0, xi), drz(t0, xi)])[np.array([0,1,2]), np.array([0,0,0])]*l.nc/l.tnc
        v_angle  = lambda x, xi: np.arccos(np.dot(v1_guess(x,xi),v0)/(norm(v0)*norm(v1_guess(x,xi))))*180/np.pi

        #------------------------------------------------------------------------
        #------------------------------------------------------------------------
        #---------------------------- Create ODE --------------------------------
        #------------------------------------------------------------------------
        #------------------------------------------------------------------------

        if 0:
            penalty = 2.0
            if  not l.clockwise:
                st.markdown('Option 1')
                res_rx = lambda x, xi: d2rx(x,xi) + mu*rx(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2)) + penalty*(v_angle(x,xi)>90)
                res_ry = lambda x, xi: d2ry(x,xi) + mu*ry(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2)) + penalty*(v_angle(x,xi)>90)
                res_rz = lambda x, xi: d2rz(x,xi) + mu*rz(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2)) + + penalty*(v_angle(x,xi)>90)

            else: 
                st.markdown('Option 2')
                res_rx = lambda x, xi: abs(d2rx(x,xi) + mu*rx(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2))) + penalty*(v_angle(x,xi)<90)
                res_ry = lambda x, xi: abs(d2ry(x,xi) + mu*ry(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2))) + penalty*(v_angle(x,xi)<90)
                res_rz = lambda x, xi: abs(d2rz(x,xi) + mu*rz(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2))) + penalty*(v_angle(x,xi)<90)

        else:
            r_norm = lambda x, xi: (rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(1/2)
            res_rx = lambda x, xi: d2rx(x,xi) + mu*rx(x,xi)/(r_norm(x,xi)**3) * ( 1.0 + Include_J2*1.5*J2*(R_sun/(r_norm(x,xi))**2)*(1-5*(rz(x,xi)/r_norm(x,xi))**2) )
            res_ry = lambda x, xi: d2ry(x,xi) + mu*ry(x,xi)/(r_norm(x,xi)**3) * ( 1.0 + Include_J2*1.5*J2*(R_sun/(r_norm(x,xi))**2)*(1-5*(rz(x,xi)/r_norm(x,xi))**2) )
            res_rz = lambda x, xi: d2rz(x,xi) + mu*rz(x,xi)/(r_norm(x,xi)**3) * ( 1.0 + Include_J2*1.5*J2*(R_sun/(r_norm(x,xi))**2)*(3-5*(rz(x,xi)/r_norm(x,xi))**2) )

        L = jit(lambda xi: np.hstack([res_rx(x, xi), res_ry(x,xi), res_rz(x,xi)]))

        #------------------------------------------------------------------------
        #------------------------------------------------------------------------
        #------------------------ Minimize the residual -------------------------
        #------------------------------------------------------------------------
        #------------------------------------------------------------------------

        # set up weights dict
        xi = TFCDict({'x':onp.zeros(H(x).shape[1]), 'y':onp.zeros(H(x).shape[1]), 'z':onp.zeros(H(x).shape[1])})
        
        # set up initial guess
        # xi['x'] = onp.dot(onp.linalg.pinv(jacfwd(rx,1)(np.array([0]),xi)['x']),r0[0:1]-rx(np.array([0]),xi))
        # xi['y'] = onp.dot(onp.linalg.pinv(jacfwd(ry,1)(np.array([0]),xi)['y']),r0[1:2]-ry(np.array([0]),xi))
        # xi['z'] = onp.dot(onp.linalg.pinv(jacfwd(rz,1)(np.array([0]),xi)['z']),r0[2:3]-rz(np.array([0]),xi))

        # Create NLLS class
        # nlls = NllsClass(xi,L,timer=True)
        # xi,_,time = nlls.run(xi)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            que = queue.Queue()

            thread1 = Thread(target=lambda q, xi, L, maxIter, method, timer: q.put(NLLS(xiInit=xi, res=L, maxIter=maxIter, method=method, timer=timer)), args = (que, xi, L, 200, method, True))
            thread1.start()
            
            seconds = 0
            progress_bar = st.empty()

            while thread1.isAlive():
                time.sleep(1)
                progress_bar.write(f'<p style="text-align:left;color:#ffffff;font-size:20px;border-radius:0%;"> \
                                    Training for {seconds}s \
                                    <p>', unsafe_allow_html=True)
                seconds +=1

            st.markdown(f'<p style="text-align:left;color:#ffffff;font-size:20px;border-radius:0%;"> \
                                Training Finished \
                                <p>', unsafe_allow_html=True)
                    
            thread1.join()
            xi, _, Time = que.get()
            runtime = time.time_ns()/(10 ** 9) - start


            st.markdown(f'<p style="text-align:left;color:#ffffff;font-size:20px;border-radius:0%;"> \
                            CPU Run time = {Time} <br>\
                            Real Run time = {runtime} \
                            <p>', unsafe_allow_html=True)

            # ic(v1_guess(t0,xi))
            # ic(v1, l.clockwise, v_angle(t0,xi), norm(L(xi)))    


            # run_time = time.time() - start

            #------------------------------------------------------------------------
            #------------------------------------------------------------------------
            #------------------- Calculate the error on the test set ----------------
            #------------------------------------------------------------------------
            #------------------------------------------------------------------------

            time_Extend = 1

            testSet = np.linspace(0,time_Extend*ub,100)
            # error = np.abs(ry(testSet,xi)-realSoln(testSet))
            predicted_velocity_x = (drx(np.array([0.0]),xi)*l.nc/l.tnc)[0]
            vel_error_x = (v1[0] - predicted_velocity_x) / v1[0]

            predicted_velocity_y = (dry(np.array([0.0]),xi)*l.nc/l.tnc)[0]
            vel_error_y = (v1[1] - predicted_velocity_y) / v1[1]

            predicted_velocity_z = (drz(np.array([0.0]),xi)*l.nc/l.tnc)[0]
            vel_error_z = (v1[2] - predicted_velocity_z) / v1[2]

            st.markdown(f'<p style="text-align:left;color:#00cf22;font-size:20px;border-radius:0%;"> \
                        initial velocity error = {vel_error_x}, {vel_error_y}, {vel_error_z}% \
                        <p>', unsafe_allow_html=True)

            st.markdown(f'<p style="text-align:left;color:#00cf22 ;font-size:20px;border-radius:0%;"> \
                                V0 desired &nbsp;&nbsp;&nbsp;  :{onp.round(v1,3)} km/s <br> \
                                V0 predicted :{onp.round([predicted_velocity_x, predicted_velocity_y, predicted_velocity_z],3)} km/s <br> \
                                relative error = {norm([vel_error_x, vel_error_y, vel_error_z]):.10f}  \
                                </p>', unsafe_allow_html=True)

            ### Calculate Residual Error ###
            Error_testSet = np.linspace(0,time_Extend*ub,1000)
            residuals_x = res_rx(Error_testSet, xi)
            residuals_y = res_ry(Error_testSet, xi)
            residuals_z = res_rz(Error_testSet, xi)
            

            #------------------------------------------------------------------------
            #------------------------------------------------------------------------
            #--------------------- Calculate Position Error  ------------------------
            #------------------------------------------------------------------------
            #------------------------------------------------------------------------
            if save_orbit:

                v1_pred = np.array([predicted_velocity_x, predicted_velocity_y, predicted_velocity_z])
                with open('OrbitParams.npy','wb') as file:
                    np.save(file, l._r0)
                    np.save(file, l._rf)
                    np.save(file, l._v0)
                    np.save(file, l._vf)

                    np.save(file, l.v1)
                    np.save(file, v1_pred)
                    
                    np.save(file, l.TOF)
                    np.save(file, l.t1)

                pk_error, TFC_error, error_ratio = Get_PropagationError()

            states, _  = plot3D_grav(r0*l.nc, v1,              ub*l.tnc, uf=rf*l.nc, J2 = J2, R = R_sun*l.nc)
            states2, _ = plot3D_grav(r0*l.nc, v1_guess(t0,xi), ub*l.tnc, uf=rf*l.nc, J2 = J2, R = R_sun*l.nc)
            

            predict_rf = states[-1,np.array([0,2,4])]
            predict_rf2 = states2[-1,np.array([0,2,4])]

            st.markdown('\n')
            # st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
            #             rf target     : {rf*l.nc} <br> \
            #             rf with PyKEP : { states[-1,np.array([0,2,4])]} <br> \
            #             rf with TFC   : { predict_rf2} \
                        # <p>', unsafe_allow_html=True)


            st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                        PyKEP final position error: {norm(predict_rf - rf*l.nc)/1000} km <br> \
                        Model final position error: {norm(predict_rf2 - rf*l.nc)/1000} km \
                        <p>', unsafe_allow_html=True)

        #------------------------------------------------------------------------
        #------------------------------------------------------------------------
        #------------------------------ Graphing --------------------------------
        #------------------------------------------------------------------------
        #------------------------------------------------------------------------ 
        
        if plot:
            plt.style.use('default')
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300

            ### Set Plot Name ###
            if Include_J2:
                if run_type=='TFC':
                    posplot_image_name = 'J2_PositionPlot'
                    poserror_image_name = 'J2_PositionError'
                    residual_image_name = 'J2_residuals'
                elif run_type=='XTFC':
                    posplot_image_name = 'J2_PositionPloXTFCt'
                    poserror_image_name = 'J2_PositionErrorXTFC'
                    residual_image_name = 'J2_residualsXTFC'
            else:
                if run_type=='TFC':
                    posplot_image_name = 'PositionPlot'
                    poserror_image_name = 'PositionError'
                    residual_image_name = 'residuals'
                elif run_type=='XTFC':
                    posplot_image_name = 'PositionPloXTFC'
                    poserror_image_name = 'PositionErrorXTFC'
                    residual_image_name = 'residualsXTFC'
            
            ### Position Plot ###
            # fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (1,1)))
            # ax = plt.gca()
            col1, col2, col3 = st.columns(3)

            states, t, _  = plot3D_grav(r0*l.nc, v1, time_Extend*ub*l.tnc, uf=rf*l.nc, J2 = J2, R = R_sun*l.nc, return_t_array=True)
            fig, ax = plt.gcf(), plt.gca()
            # states, t, _  = plot3D_grav(r0*l.nc, v1, time_Extend*ub*l.tnc, uf=rf*l.nc, ax=ax, J2 = J2, R = R_sun*l.nc, return_t_array=True)

            ax.plot(rx(testSet,xi)*l.nc/1000,ry(testSet,xi)*l.nc/1000, 'ro', markersize=3 ,label='TFC Solution')
            format_axes(ax=ax, fontsize=12, xlabel = r'$R_{x}$ [km]', ylabel=r'$R_{y}$ [km]')
            # fig.savefig('./Plot/'+posplot_image_name+'.pdf', bbox_inches='tight')

            with col1:
                # avoid fading when reloading - see link https://tinyurl.com/3a3u3waf
                st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)
                st.pyplot(fig)
            

            ### Position Error ###
            pos_x = rx(t/l.tnc, xi)*l.nc
            pos_y = ry(t/l.tnc, xi)*l.nc
            pos_z = rz(t/l.tnc, xi)*l.nc
            x_error = (pos_x - states[:,0])/states[:,0]
            y_error = (pos_y - states[:,2])/states[:,2]
            z_error = (pos_z - states[:,4])/states[:,4]

            fig2 = plt.figure(figsize=set_size(483.69687*1.05, 1))
            ax = plt.gca()
            ax.semilogy(t/l.tnc/ub, onp.abs(x_error), 'bo', label=r'$R_{x}$ error', markersize=2, markeredgewidth=0.01)
            ax.semilogy(t/l.tnc/ub, onp.abs(y_error), 'go', label=r'$R_{y}$ error', markersize=2, markeredgewidth=0.01)
            ax.semilogy(t/l.tnc/ub, onp.abs(z_error), 'ro', label=r'$R_{z}$ error', markersize=2, markeredgewidth=0.01)
            format_axes(ax=ax, fontsize=15, xlabel = 'time [ND]', ylabel=r'Relative Error Magnitude', scale_legend=True)
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
            # fig2.savefig('./Plot/'+poserror_image_name+'.pdf', bbox_inches='tight')

            with col2:
                # avoid fading when reloading - see link https://tinyurl.com/3a3u3waf
                st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)
                st.pyplot(fig2)

            ### Residual Error ###

            fig3 = plt.figure(figsize=set_size(483.69687*1.05, 1))
            ax = plt.gca()
            ax.semilogy(Error_testSet/ub, (onp.abs(residuals_x)), 'bo', label='x residual', markersize=2, markeredgewidth=0.01)
            # ax.semilogy(Error_testSet/ub, (onp.abs(residuals_x)), 'yx', markersize=3, markeredgewidth=0.01)

            ax.semilogy(Error_testSet/ub, (onp.abs(residuals_y)), 'go', label='y residual', markersize=2,  markeredgewidth=0.01)
            # ax.semilogy(Error_testSet/ub, (onp.abs(residuals_y)), 'gx', markersize=3, markeredgewidth=0.01)

            ax.semilogy(Error_testSet/ub, (onp.abs(residuals_z)), 'ro', label='z residual', markersize=2,  markeredgewidth=0.1)
            # ax.semilogy(Error_testSet/ub, (onp.abs(residuals_z)), 'rx', markersize=3,  markeredgewidth=0.1)
            # Show the major grid and style it slightly.
            ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
            format_axes(ax=ax, fontsize=15, xlabel = 'time [ND]', ylabel=r'Absolute Error Magnitude', scale_legend=True)

            # fig3.savefig('./Plot/'+residual_image_name+'.pdf', bbox_inches='tight')
            with col3:
                # avoid fading when reloading - see link https://tinyurl.com/3a3u3waf
                st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)
                st.pyplot(fig3)
            

        if Include_J2:
            return [pk_error, TFC_error, error_ratio], runtime
        else:
            return float(norm([vel_error_x, vel_error_y, vel_error_z])), runtime



def train_models(l, poly_removes=[2], methods = ["pinv"], plot=False, save_orbit=False, run_type='TFC', inputs={}):
    data = []
    total_start = time.time_ns()/(10 ** 9)

    Training_indicator = st.empty()
    col1_a, col2_a, col3_a = st.columns([1,2,1])
    placeholder = st.empty()


    for way in [True]:

        for poly_order in inputs['Order']:
            for poly_remove in poly_removes:
                for basis_func in inputs['Polynomial']:
                    for method in methods:
                        for point in inputs['Points']:
                            for i in range(inputs.get('num_runs')): 
                                with placeholder.container(): col1, col2, col3 = st.columns([1,2,1])
                                error, runtime = TrainModel(l=l, 
                                                            points=point, 
                                                            poly_order=poly_order, 
                                                            poly_removed=poly_remove, 
                                                            basis_func=basis_func, 
                                                            method = method, 
                                                            plot=plot, 
                                                            save_orbit=save_orbit,
                                                            run_type=run_type, 
                                                            inputs=inputs,
                                                            container=placeholder)
                                if inputs['Include_J2']:
                                    data.append([way, poly_order, poly_remove, basis_func,  method, i, error[0], error[1], error[2], runtime])
                                else:
                                    data.append([way, poly_order, poly_remove, basis_func,  method, i, error, runtime])
                                plt.close('all')

                                with col2_a: st.markdown(f'<p style="text-align:left;color:#ed9213 ;font-size:20px;border-radius:0%;"> \
                                                            Trained way={way}, Order={poly_order}, removed={poly_remove}, basis={basis_func}, method={method}, num points={point, i}\
                                                            <p>', unsafe_allow_html=True)



    total_runtime = time.time_ns()/(10 ** 9) - total_start
    with col2_a: st.markdown(f'<p style="text-align:left;color:#ffffff ;font-size:20px;border-radius:0%;"> \
                            Traing took {total_runtime} seconds\
                            <p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])


    if inputs['Include_J2']:
        TrainingDf = pd.DataFrame(data, columns=['Shortway', 'poly_order', 'poly_remove', 'basis_function', 'method', 'Example', 'PK Loss', 'TFC Loss', 'Loss Ratio','Training Time'])

        with col2: st.table(TrainingDf)

        TrainingStats = TrainingDf.groupby(['Shortway', 'basis_function', 'method', 'poly_order', 'poly_remove'])[['PK Loss', 'TFC Loss', 'Loss Ratio','Training Time']].median()

        TrainingStats['PK Loss'] = TrainingStats['PK Loss'].map(lambda x: '%.5e' % x)
        TrainingStats['TFC Loss'] = TrainingStats['TFC Loss'].map(lambda x: '%.5e' % x)
        TrainingStats['Loss Ratio'] = TrainingStats['Loss Ratio'].map(lambda x: '%.5e' % x)

        with col2: st.table(TrainingStats)
        # TrainingStats.to_pickle("Training_DF")

    else:

        TrainingDf = pd.DataFrame(data, columns=['Shortway', 'poly_order', 'poly_remove', 'basis_function', 'method', 'Example', 'Loss', 'Training Time'])
        TrainingDf['Passed 1e-10'] = TrainingDf['Loss'].map(lambda x: 1.0*(x<1e-10))
        TrainingDf['Passed 1e-12'] = TrainingDf['Loss'].map(lambda x: 1.0*(x<1e-12))
        TrainingDf['Passed 1e-13'] = TrainingDf['Loss'].map(lambda x: 1.0*(x<1e-13))

        with col2: st.table(TrainingDf)

        TrainingStats = TrainingDf.groupby(['Shortway', 'basis_function', 'method', 'poly_order', 'poly_remove'])['Loss', 'Training Time'].median()

        TrainingStats[['Percent 1e-10',
                    'Percent 1e-12', 
                    'Percent 1e-13']] = TrainingDf.groupby(['Shortway', \
                                                            'basis_function', \
                                                            'method', \
                                                            'poly_order', \
                                                            'poly_remove'])[['Passed 1e-10', 'Passed 1e-12', 'Passed 1e-13']].mean()*100.0
        # TrainingStats[['Percent 10', 'Percent 0.1', 'Percent 0.001']] = TrainingStats[['Percent 10', 'Percent 0.1', 'Percent 0.001']].apply(lambda x: '%.2f' % x)
        TrainingStats['Loss'] = TrainingStats['Loss'].map(lambda x: '%.5e' % x)

        with col2: st.table(TrainingStats)
        # TrainingStats.to_pickle("Training_DF")

    return TrainingDf, TrainingStats

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#--------------------- Execute single TFC or XTFC run -------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
if __name__ == '__main__':

    if 1:
        TrainingDf, TrainingStats = train_models(save_orbit=True, plot=True, run_type='TFC')
    if 0:
        TrainingDf, TrainingStats = train_models(points=[200], poly_orders=[100], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"],  save_orbit=True, plot=True, run_type='XTFC')


    # In[1]:
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #---------------------- Sweep over configurations -----------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    # 
    ### Sweep over random seeds ### 
    if 0:

        TrainingDf, TrainingStats = train_models(points=[200], poly_orders=[100], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"],  save_orbit=True, plot=False)
        bins = onp.logspace(-11, -4, 18)

        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (1,1)))
        ax = plt.gca()
        ax.hist(TrainingDf['Loss'], bins)
        ax.set_xscale('log')
        format_axes(ax=ax, fontsize=12, xlabel = r'Relative Error Magnitude', ylabel=r'Count', force_ticks=bins)
        fig.savefig('./Plot/Seed_Error.pdf')

    ### Sweep over polynomial orders ###
    if 0:
        poly_orders=list(range(1,200, 2))
        TrainingDf, TrainingStats = train_models(poly_orders=poly_orders)

        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(poly_orders, TrainingDf['Loss'], 'b+', markersize=5, markeredgewidth=0.01)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order', ylabel=r'Relative Error Magnitude', scale_legend=True)
        fig.savefig('./Plot/poly_orders.pdf', bbox_inches='tight')

    ### Sweep over removed bias functions ###
    if 0:
        poly_removes=list(range(-1,10, 1))
        TrainingDf, TrainingStats = train_models(poly_removes=poly_removes)

        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(poly_removes, TrainingDf['Loss'], 'b+', markersize=15, markeredgewidth=0.01)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order removed', ylabel=r'Relative Error Magnitude', scale_legend=True, force_ticks=poly_removes)
        fig.savefig('./Plot/poly_removes.pdf', bbox_inches='tight')

    ### Sweep over number of training points ###
    if 0:
        points=list(range(1,200, 1))
        TrainingDf, TrainingStats = train_models(points=points)
        
        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(points, TrainingDf['Loss'], 'b+', markersize=5 , markeredgewidth=0.01)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'Number of Training Points ', ylabel=r'Relative Error Magnitude', scale_legend=True)
        fig.savefig('./Plot/Points.pdf', bbox_inches='tight')


    ############ X-TFC ###############################

    ### Sweep over polynomial orders ###
    if 0:
        poly_orders=list(range(1,200, 2))
        TrainingDf, TrainingStats = train_models(points=[100], poly_orders=poly_orders, poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"])

        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(poly_orders, TrainingDf['Loss'], 'b+', markersize=5, markeredgewidth=0.01)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'Number of Neurons', ylabel=r'Relative Error Magnitude', scale_legend=True)
        fig.savefig('./Plot/NumNeurons.pdf', bbox_inches='tight')

    ### Sweep over removed bias functions ###
    if 0:
        poly_removes=list(range(-1,10, 1))
        TrainingDf, TrainingStats = train_models(points=[100], poly_orders=[50], poly_removes=poly_removes, basis_funcs=['ELMTanh'], methods = ["lstsq"])

        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(poly_removes, TrainingDf['Loss'], 'b+', markersize=15, markeredgewidth=0.01)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order removed', ylabel=r'Relative Error Magnitude', scale_legend=True, force_ticks=poly_removes)
        fig.savefig('./Plot/poly_removesXTFC.pdf', bbox_inches='tight')

    ### Sweep over number of training points ###
    if 0:
        points=list(range(1,200, 1))
        TrainingDf, TrainingStats = train_models(points=points, poly_orders=[50], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"])
        
        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(points, TrainingDf['Loss'], 'b+', markersize=5 , markeredgewidth=0.01)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'Number of Training Points ', ylabel=r'Relative Error Magnitude', scale_legend=True)
        fig.savefig('./Plot/PointsXTFC.pdf', bbox_inches='tight')

