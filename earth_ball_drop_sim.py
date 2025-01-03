# -*- coding: utf-8 -*-
"""
@author: me
"""


import numpy as np
import matplotlib.pyplot as plt

from math import pi

# solve_ivp is an numerical ODE solver
from scipy.integrate import solve_ivp

from scipy.interpolate import PchipInterpolator



# Define constants pseudo-globally through class
# definition. Effectively a dictionary
class Constants_Earth:
    r_surface = 6378.0e3
    azimuthal_angular_velocity_earth = 2 * pi / (24 * 60 * 60)
    earth_mass = 5.972e24 # kg
    G = 6.6743e-11
    angular_velocity_vector = np.array([0, 
                                        0, 
                                        azimuthal_angular_velocity_earth])
    
    # Other relevant info
    r_geostationary = 35786*1e3 # metres
#end Constants_Earth



def Gravity_accn_(t, u):
    x = u[0]
    y = u[1]
    z = u[2]
    
    G = Constants_Earth.G
    M = Constants_Earth.earth_mass
    r_sqr_ = x*x + y*y + z*z
    
    accn_magnitude = - G * M / r_sqr_
    
    vector_magnitude = np.sqrt(r_sqr_)
    vec_x_proportion = x / vector_magnitude
    vec_y_proportion = y / vector_magnitude
    vec_z_proportion = z / vector_magnitude
    
    accn_x = vec_x_proportion * accn_magnitude
    accn_y = vec_y_proportion * accn_magnitude
    accn_z = vec_z_proportion * accn_magnitude
    
    return [accn_x, accn_y, accn_z]
  
  
def Aero_drag_accn_(t, u, mass_of_object = 1):
    # unimplemented for now, but you can
    # code something into this,
    # e.g., make altitude calc, or fake reentry
    return [0, 0, 0]


def Drop_ode(t, u):
    """Here we define a ODE in first-order form:
        du/dt = F(t, u)
    where u and F can be vectors.
        u is our "state space"
        t is time
        
    We use this form as there a mnay first order ODE numerical
    solvers available.
    
    Mathematically we transform our ODE of
        m*a = F_gravity    (GOV.EQ)
    
    by defining:
        u_1 := x
        u_2 := y
        u_3 := z
    however, we also define
        u_4 := v_x (velocity in x)
        u_5 := v_y 
        u_6 := v_z
        
    Next, you simply compute what du/dt should be, e.g.
        d(u_1)/dt = d( x )/dt = v_x
        [which is]= d( u_4 )/dt
    and for an acceleration term, like a_x:
        d(u_4)/dt = d( v_x ) / dt
                  = a_x 
    [ which from our GOV.EQ we get]
                  = (F_gravity_x) / m
    [ so we can re-write this as]
        d(u_4)/dt = F_gravity_x / m
        
    as an additional simplification we just set m=1, as we primarily
    care about accelerations, and no other forces are present (yet)
    """
    # Let
    x = u[0]
    y = u[1]
    z = u[2]
    
    # and velocities:
    vx = u[3]
    vy = u[4]
    vz = u[5]
    
    dudt = [None] * 6; #pre-allocate list
    dudt[0] = vx
    dudt[1] = vy
    dudt[2] = vz
    
    # calc gravity (x, y) components:
    f_grav_vector = Gravity_accn_(t, u)
    # this aero_drag assumes a mass of 1 by default, however
    # it just returns a (0,0,0) vector. 
    aero_drag_vector = Aero_drag_accn_(t, u);
    
    # acceleration in x
    dudt[3] = f_grav_vector[0] + aero_drag_vector[0]
    # acceleration in y
    dudt[4] = f_grav_vector[1] + aero_drag_vector[1]
    # ... in z
    dudt[5] = f_grav_vector[2] + aero_drag_vector[2]
    
    return dudt


def Surface_reached_event_(t, u):
    # Event detection
    # Check that x^2 + y^2 <= r_surf^2
    
    x = u[0]
    y = u[1]
    z = u[2]
    
    r_squared_ = x*x + y*y + z*z
    r_surf_squared = Constants_Earth.r_surface \
                   * Constants_Earth.r_surface
    
    return r_squared_ - r_surf_squared
# Add decorators required by `solve_ivp` to stop solving if
# ball hits the surface:
Surface_reached_event_.terminal = True
Surface_reached_event_.direction = -1


# ----------------------------------------------------------------------------
# Spherical-Cartesian coordinate conversions (might be wrong)
def Radius_from_cart(u):
    x = u[0,:]
    y = u[1,:]
    z = u[2,:]
    return np.sqrt(x*x + y*y + z*z)


def Azi_angle_calc(u, r_vec = None):
    x = u[0,:]
    y = u[1,:]
    z = u[2,:]
    
    # using WIKI: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    azi = np.sign(y) * np.arccos(x / np.sqrt(x*x + y*y))
    
    b_ = x>0
    azi[b_] = np.arctan2(y[b_],x[b_])
    
    b_ = np.logical_and(x<0, y>=0)
    azi[b_] = np.arctan2(y[b_],x[b_]) + pi
    
    b_ = np.logical_and(x<0 , y<0)
    azi[b_] =np.arctan2(y[b_],x[b_]) - pi
    
    azi[np.logical_and(x==0 , y>0)] = pi * 0.5
    azi[np.logical_and(x==0 , y<0)] = - pi * 0.5
    # edge case of 0,0 not expected
    return azi


def pol_angle_calc(u, r_vec):
    # x = u[0,:]
    # y = u[1,:]
    z = u[2,:]
    
    # using WIKI: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    pol = np.arccos(z / r_vec)
    
    return pol


def Sphere_to_cart(r, polar_angle, azi_angle, size):
    # poor `return` implementation; sorry about this - written quickly
    X = np.zeros((3, size))
    
    
    X[0,:] = r * np.sin(polar_angle) * np.cos(azi_angle)
    X[1,:] = r * np.sin(polar_angle) * np.sin(azi_angle)
    X[2,:] = r * np.cos(polar_angle)
    
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    return x,y,z


def Plot_sphere(list_center, list_radius, ax = None, name = "Earth"):
    # Copied / modified from:
    # https://stackoverflow.com/questions/64656951/plotting-spheres-of-radius-r
    if (ax == None):
        ax = fig.add_subplot(projection='3d')
    #
    
    for c, r in zip(list_center, list_radius):

        # draw sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
    
        ax.plot_surface(x-c[0],
                        y-c[1], 
                        z-c[2], 
                        color = 'b', 
                        alpha = 0.9, 
                        linewidth = 2.5)
    return


def Calc_surface_azimuth_displacement(t, azi_initial):
    return (Constants_Earth.azimuthal_angular_velocity_earth * sol.t \
                + azi_angle_ini)

# ============================================================================
if __name__ == "__main__":
    # main code here
    enable_user_input = True # Turns on text entry through terminal
    SET_SCALE_EQUAL = True   # Set equal axis scaling
    PLOT_SURFACE_TRACE = True # Plot surface projection of ball on earth
    PLOT_EARTH_ALWAYS = False # False: only plot earth if ladder height > 1e6
    
    
    # User info: 
    if (enable_user_input):
        ladder_height = abs(float(input("> Enter ladder height [m] : ")))
        print(" (polar angle = 0 -- North Pole)")
        print(" (polar angle = 90 -- equator)")
        print(" (recommended angle in [0, 90]; not tested beyond that)")
        polar_angle_ini = np.deg2rad(
                                     float(
                                        input("> Enter polar angle [deg] : ")
                                          )
                                    )
    else:
        ladder_height = Constants_Earth.r_geostationary * 0.1 # metres
        polar_angle_ini = 0.34*pi/2 # 0 for pole; pi/2 for equator
    
    
    # -----------------------------------------------------------------------
    azi_angle_ini = 0 # leave as zero for simplicity
    
    # mass only needed IF you implement drag forces
    mass_of_ball = 1
    
    # have fun with this
    user_throw_vel_x = 0
    user_throw_vel_y = 0 # 5.220e3 # <- an "almost orbit" when at 0.1*r_geostat
    user_throw_vel_z = 0


    # short hand these
    r_surface       = Constants_Earth.r_surface# metres
    azi_angular_vel = Constants_Earth.azimuthal_angular_velocity_earth # rad/s


    # Initial condition calculation ------------------------------------------
    # Positions (spherical to cartesian)
    r_ini_ = r_surface + ladder_height
    
    x_ini, y_ini, z_ini = Sphere_to_cart(r_ini_, 
                                         polar_angle_ini, 
                                         azi_angle_ini,
                                         1)
    
    x_ini = x_ini[0]
    y_ini = y_ini[0]
    z_ini = z_ini[0]
    
    # Get vector magnitude for later calcs
    norm_ = np.linalg.norm([x_ini, y_ini, z_ini])
    pos_vec_ = np.array([x_ini, y_ini, z_ini ])
    
    
    initial_vel_via_rotation = np.cross(Constants_Earth.angular_velocity_vector, 
                                        pos_vec_)
    
    
    
    # Direction of initial release velocity is in direction of
    # of `y` (assumes perfectly rigid ladder)
    vx_ini = 0 + initial_vel_via_rotation[0] + user_throw_vel_x
    vy_ini = 0 + initial_vel_via_rotation[1] + user_throw_vel_y
    vz_ini = 0 + initial_vel_via_rotation[2] + user_throw_vel_z
    
    
    # declare initial state_space vector
    u_initial = [x_ini, 
                 y_ini,
                 z_ini,
                 vx_ini,
                 vy_ini,
                 vz_ini]
    
    time_span = [0, 5e4] # in seconds
    
    # Use solve_ivp to numerically solve the mechanics ODE problem
    sol = solve_ivp(Drop_ode, 
                    time_span, 
                    u_initial, 
                    method = 'LSODA', 
                    events = Surface_reached_event_,
                    rtol = 1e-8,
                    atol = 1e-8)
    
    # Surface trace of initial position (on surface)
    azi_surf__ = Calc_surface_azimuth_displacement(sol.t, azi_angle_ini)
    
    # Calculate x,y,z of surface trace for plotting        
    x_surf, y_surf, z_surf = Sphere_to_cart(r_surface, 
                                         polar_angle_ini, 
                                         azi_surf__,
                                         azi_surf__.size)
    
    # Handle cases where event has or has not occured
    if (sol.t_events == sol.t[-1]):
        xs_event = x_surf[-1]
        ys_event = y_surf[-1]
        zs_event = z_surf[-1]
        EVENT_OCCURED_FLAG = True
    elif (sol.t_events[0].size < 1):
        print("No events occured")
        EVENT_OCCURED_FLAG = False
    else:
        # if t_event is not in sol.t, compute manually
        azi__ = Calc_surface_azimuth_displacement(sol.t_events, azi_angle_ini)
        
        xs_event, ys_event, zs_event = Sphere_to_cart(r_surface, 
                                                      polar_angle_ini, 
                                                      azi__,
                                                      1)
        EVENT_OCCURED_FLAG = True
    # end if    

    # Short hand notations for solution
    t = sol.t
    x = sol.y[0,:]
    y = sol.y[1,:]
    z = sol.y[2,:]
    
    # (`obs` means 'observed', i.e. projected)
    r_obs_ = Radius_from_cart(sol.y)
    azi_obs_ = Azi_angle_calc(sol.y, r_obs_)
    pol_obs_ = pol_angle_calc(sol.y, r_obs_)
    
    
    # Plotting ---------------------------------------------------------------
    fig = plt.figure()
    fig.set_figwidth(15)
    
    if ((ladder_height > 1e6) or PLOT_EARTH_ALWAYS):
        PLOT_EARTH = True
    else:
        PLOT_EARTH = False
    
    if (True):
        ax = fig.add_subplot(121,projection='3d')
        
        if (PLOT_EARTH):
            # retain axis limits from before
            xl__ = [min(x)*0.7, max(x) * 1.3]#ax.get_xlim()
            yl__ = [min(y)*0.7, max(y) * 1.3]#ax.get_ylim()
            zl__ = [min(z)*0.7, max(z) * 1.3]#ax.get_ylim()
            
            # make sphere smaller to enable plot clipping
            Plot_sphere([(0,0,0)], [r_surface * 0.88,], ax = ax)
            ax.set_xlim(xl__[0], xl__[1])
            ax.set_ylim(yl__[0], yl__[1])
            ax.set_zlim(zl__[0], zl__[1])
        
        cmap_ = 'hsv'
        ax.set_zlabel("z")
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        sc = ax.scatter(x, 
                        y, 
                        z, 
                        c = t, 
                        marker = 'o', 
                        cmap = cmap_,
                        label = "Ball (dropped)"
                        )#gist_ncar
        plt.colorbar(sc)
            
        if (PLOT_SURFACE_TRACE):
            ax.scatter(x_surf, 
                       y_surf, 
                       z_surf, 
                       c = t, 
                       s = 120, 
                       marker = '+', 
                       cmap = cmap_,
                       label = "Ladder base on Earth's surface")
            
            #x_, y_, z_ = Sphere_to_cart(r_surface, pol_obs_, azi_obs_, azi_obs_.size)
            #ax.plot(x_, y_, z_, '-', color = 'r')
        #end if
        
        if (SET_SCALE_EQUAL):
            ax.set_aspect('equal', adjustable='box')
        #end if
        
        ax.legend(loc = 'best')
    #end if
    
    
    if (True):
        #fig = plt.figure()
        ax2 = fig.add_subplot(122)#(projection='3d')
        ax2.plot(sol.t, r_obs_ - r_surface, '.-')
        ax2.set_ylabel('$R-R_\mathrm{surface}$ [m]')
        ax2.set_xlabel('Time $t$')
        ax2.grid(visible=True, which='major', axis='both')
        ax2.grid(visible=True, which='minor', axis='both', linestyle = ':')
        
                 # print distance at end time (Euclidean; change to surface distance!)
        if (EVENT_OCCURED_FLAG):
            xintr = PchipInterpolator(sol.t, x_surf)
            yintr = PchipInterpolator(sol.t, y_surf)
            zintr = PchipInterpolator(sol.t, z_surf)
            
            t_ev = sol.t_events[0]
            surf_vec = np.array([xintr(t_ev), yintr(t_ev), zintr(t_ev)]).flatten()
            
            dist_ = np.linalg.norm(sol.y_events[0][0,:3] - surf_vec)
            print(f"\n\nThe Euclidean distance is {dist_:.3f} m")
            ax2.set_title("The final Euclidean distance"
                          " $| \mathbf{x}_\mathrm{landed} - "
                          "\mathbf{x}_\mathrm{ladder\;base} |$ "
                          f"is {dist_:.5e} m")
        #end if
    #end if
#end main
