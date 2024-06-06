# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:52:50 2024

@author: apironet
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 

def Calcium_la(t,period):
    # CALCIUM_LA Calcium input function for the left atrium
    # Curve from Nygren98, used in Shim08 with a period of 0.95 s
    
    T1 = 0.0178 
    T2 = 0.0632 
    maxCa = 1.33 
    
    t = np.mod(t*0.95/period,0.95) 
    
    Ca = 0.5*maxCa*((1 - np.cos(t*np.pi/T1))*(t >= 0)*(t <= T1) + (1 + np.cos((t - T1)*np.pi/(T2 - T1)))*(t > T1)*(t <= T2)) 

    return Ca

def Calcium_lv(t,period):
    # CALCIUM_LV Calcium input function for the left ventricle
    # Curve from Yue87, used in Burkhoff94 with a period of 0.6 s
    
    T1 = 0.0406 
    T2 = 0.1302 
    maxCa = 1.47 
    
    t = np.mod(t*0.6/period,0.6) 
    
    Ca = 0.5*maxCa*((1 - np.cos(t*np.pi/T1))*(t >= 0)*(t <= T1) + (1 + np.cos((t - T1)*np.pi/(T2 - T1)))*(t > T1)*(t <= T2)) 

    return Ca

def driver_rv(t,period):
    # DRIVER_RV Driver function for the right ventricle
    # Curve from Chung96, used with a period of 0.6 s
    
    Ab = [0.9556, 0.6249, 0.018] 
    Bb = [255.4, 225.3, 4225.0] 
    Cb = [0.306, 0.2026, 0.2491] 
    shift = -0.055 # + : driver shifts to the right
    
    tmod = np.mod((t - shift)*0.6/period,0.6) 
    
    e = Ab[0]*np.exp(-Bb[0]*(tmod - Cb[0])**2) + Ab[1]*np.exp(-Bb[1]*(tmod - Cb[1])**2) + Ab[2]*np.exp(-Bb[2]*(tmod - Cb[2])**2) 
    
    return e

def state_eqs(t,y,params):
    #STATE_EQS Model state equations
    
    Y1 = params[0] 
    Z1 = params[1] 
    Y2 = params[2] 
    Z2 = params[3] 
    Y3 = params[4]  
    Z3 = params[5]  
    Y4 = params[6] 
    Yd = params[7] 
    Tt = params[8] 
    B = params[9] 
    hc = params[10] 
    La = params[11] 
    R = params[12] 
    L0 = params[13]
    
    A = params[14] 
    B_s = params[15] 
    h = params[16] 
    Rlv0 = params[17] 
    
    A_la = params[18] 
    B_s_la = params[19] 
    h_la = params[20] 
    Rla0 = params[21] 
    
    Rmt = params[22] # Resistance of the mitral valve
    Rav = params[23] # Resistance of the aortic valce
    Rsys = params[24] # Systemic vascular resistance
    Rpu = params[25]
    Eao = params[26] # Aortic elstance
    Evc = params[27] # Elastance of the vena cava
    Epu = params[28] # Elastance of the pulmonary vein
    Epa = params[29] # Elastance of the pulmonary artery
    Erv = params[30] # End-systolic elastance of the right ventricle
    Rtc = params[31] # Resistance of the tricuspid valve
    Rpul = params[32] # Pulmonary vascular resistance
    Rpv = params[33] # Resistance of the pulmonary valve
    
    period = params[34] # Cardiac period
    shift = params[35]
    
    TCa = y[0] 
    TCa_star = y[1] 
    T_star = y[2] 
    X = y[3] 
    TCa_la = y[4] 
    TCa_star_la = y[5] 
    T_star_la = y[6]
    X_la = y[7]
    Vlv = y[8] # Volume of the left ventricle
    Vao = y[9] # Volume of the aorta
    Vla = y[10] # Volume of the left atrium
    Vvc = y[11] # Volume of the vena cava
    Vrv = y[12] # Volume of the right ventricle
    Vpa = y[13] # Volume of the pulmonary artery
    Vpu = y[14] # Volume of the pulmonary vein
    
    # Left ventricle
    
    Rlv = (Vlv*3/(2*np.pi))**(1/3) 
    L = L0*Rlv/Rlv0 
    
    TCa_eff = TCa*np.exp(-R*(L - La)**2) 
    
    T = Tt - y[0] - y[1] - y[2] 
    Ca = Calcium_lv(t,period) 
    
    dy = np.zeros(15)
    dy[3] = B*(L - X - hc) # dX/dt
    
    Qb = Y1*Ca*T - Z1*TCa 
    Qa = Y2*TCa_eff - Z2*TCa_star 
    Qr = Y3*TCa_star - Z3*T_star*Ca 
    Qd = Y4*T_star 
    Qd1 = Yd*(dy[3]**2)*T_star 
    Qd2 = Yd*(dy[3]**2)*TCa_star 
    
    dy[0] = Qb - Qa # dTCa/dt
    dy[1] = Qa - Qr - Qd2 # dTCa*/dt
    dy[2] = Qr - Qd - Qd1 # dT*/dt
    
    # Left atrium
    
    Rla = (Vla*3/(2*np.pi))**(1/3) 
    L_la = L0*Rla/Rla0 
    
    TCa_eff_la = TCa_la*np.exp(-R*(L_la - La)**2) 
    
    T_la = Tt - y[4] - y[5] - y[6]
    Ca_la = Calcium_la(t + shift,period) 
    
    dy[7] = B*(L_la - X_la - hc) # dX/dt
    
    Qb_la = Y1*Ca_la*T_la - Z1*TCa_la 
    Qa_la = Y2*TCa_eff_la - Z2*TCa_star_la 
    Qr_la = Y3*TCa_star_la - Z3*T_star_la*Ca_la 
    Qd_la = Y4*T_star_la 
    Qd1_la = Yd*(dy[7]**2)*T_star_la
    Qd2_la = Yd*(dy[7]**2)*TCa_star_la 
    
    dy[4] = Qb_la - Qa_la # dTCa/dt
    dy[5] = Qa_la - Qr_la - Qd2_la # dTCa*/dt
    dy[6] = Qr_la - Qd_la - Qd1_la # dT*/dt
    
    # Circulatory model
    
    Fb = A*(TCa_star + T_star)*(L - X) 
    Fp = -B_s*(1 - L/L0) 
    F = Fb + Fp 
    
    Fb_la = A_la*(TCa_star_la + T_star_la)*(L_la - X_la) 
    Fp_la = -B_s_la*(1 - L_la/L0) 
    F_la = Fb_la + Fp_la 
    
    Plv = 2*F*h/Rlv*7.5 
    Pla = 2*F_la*h_la/Rla*7.5 
    Pao = Eao*Vao 
    Pvc = Evc*Vvc 
    Prv = Erv*driver_rv(t,period)*Vrv 
    Ppa = Epa*Vpa 
    Ppu = Epu*Vpu 
    
    Qpu = (Ppu - Pla)/Rpu 
    Qmt = (Pla - Plv)/Rmt 
    Qav = (Plv - Pao)/Rav 
    Qsys = (Pao - Pvc)/Rsys 
    Qtc = (Pvc - Prv)/Rtc 
    Qpv = (Prv - Ppa)/Rpv 
    Qpul = (Ppa - Ppu)/Rpul 
    
    dy[8] = Qmt*(Qmt > 0) - Qav*(Qav > 0) # dVlv/dt
    dy[9] = Qav*(Qav > 0) - Qsys # dVao/dt
    dy[10] = Qpu - Qmt*(Qmt > 0) # dVla/dt
    dy[11] = Qsys - Qtc*(Qtc > 0) # dVvc/dt
    dy[12] = Qtc*(Qtc > 0) - Qpv*(Qpv > 0) # dVrv/dt
    dy[13] = Qpv*(Qpv > 0) - Qpul # dVpa/dt
    dy[14] = Qpul - Qmt*(Qmt > 0) # dVpu/dt
    
    return dy

# Set up parameters
    
Y1 = 39 # uM/s
Z1 = 30 # /s
Y2 = 1.3 # /s
Z2 = 1.3 # /s
Y3 = 30 # /s
Z3 = 1560 # uM/s
Y4 = 40 # /s
Yd = 9 # s/uM^2
Tt = 70 # uM
B = 1200 # /s
hc = 0.005 # um
La = 1.17 # um
R = 20 # /um^2
L0 = 0.97 # um

A = 944.5815 # mN/mm^2/um/uM (adjusted)
B_s = 0.4853 # mN/mm^2 (adjusted)
h = 3.0843/B_s # cm (Kass89) 
Rlv0 = 1.62 # cm (Kass89)

A_la = 577.81 # mN/mm^2/um/uM (adjusted)
B_s_la = 20.002 # mN/mm^2 (adjusted)
h_la = 39.6012/B_s_la # cm (Gare01)
Rla0 = 1.6611 # cm (Gare01)

Rmt = 0.0278 # (Gare01 and Kass89) 
Rav = 0.0849 # (Gare01 and Kass89)
Rsys = 3.6144 # (Gare01, Maughan87 and Kass89)
Rpu = 0.1079 # (Gare01 and Kass89)
Eao = 6.9429 # (Gare01, Maughan87 and Kass89)
Evc = 1.3077 # optimized
Epu = 0.0881 # optimized
Erv = 2.1018 # optimized
Epa = 2.2948 # approx from Maughan87
Rtc = 0.2786 # optimized
Rpv = 0.03 # (Revie11)
Rpul = 2.454 # optimized

period = 0.45 # s (Gare01)
shift = 0.085 # shift between LA and LV calcium peaks

params = [Y1, Z1, Y2, Z2, Y3, Z3, Y4, Yd, Tt, B,
          hc, La, R, L0, A, B_s, h, Rlv0, A_la, B_s_la,
          h_la, Rla0, Rmt, Rav, Rsys, Rpu, Eao, Evc, Epu, Epa,
          Erv, Rtc, Rpul, Rpv, period, shift] 

tmax = 99.9 
step = 0.001

v0 = np.array([0, 0, 0, (1.05 - hc), 0, 0, 0, (1.05 - hc), 20, 15, 10, 1, 40, 15, 172])
sol = solve_ivp(state_eqs, [0, tmax], v0, args = [params], 
                t_eval = np.arange(0, tmax, step), rtol = 1e-5, 
                max_step = 0.001)

t = sol.t
v = sol.y
 
# Left ventricle

Vlv = v[8,:] 
Rlv = (Vlv*3/(2*np.pi))**(1/3) 
L = L0*Rlv/Rlv0 

TCa = v[0,:] 
TCa_star = v[1,:] 
T_star = v[2,:]
T = Tt - v[0,:] - v[1,:] - v[2,:]
Ca = Calcium_lv(t,period) 
X = v[3,:]

Fb = A*(TCa_star + T_star)*(L - X) 
Fp = -B_s*(1 - L/L0) 
F = Fb + Fp 
Plv = 2*F*h/Rlv*7.5 

# Left Atrium

Vla = v[10,:] 
Rla = (Vla*3/(2*np.pi))**(1/3) 
L_la = L0*Rla/Rla0 

TCa_la = v[4,:] 
TCa_star_la = v[5,:]
T_star_la = v[6,:]
T_la = Tt - v[4,:] - v[5,:] - v[6,:]
Ca_la = Calcium_la(t + shift,period) 
X_la = v[7,:]

Fb_la = A_la*(TCa_star_la + T_star_la)*(L_la - X_la) 
Fp_la = -B_s_la*(1 - L_la/L0) 
F_la = Fb_la + Fp_la 
Pla = 2*F_la*h_la/Rla*7.5 

# Circulatory elements

Vao = v[9,:]
Vvc = v[11,:]
Vrv = v[12,:]
Vpa = v[13,:]
Vpu = v[14,:]

Pao = Eao*Vao 
Pvc = Evc*Vvc 
Prv = Erv*driver_rv(t,period)*Vrv 
Ppa = Epa*Vpa 
Ppu = Epu*Vpu 

plt.subplots()
plt.plot(t, Pao)
plt.plot(t, Plv)
plt.plot(t, Pla)

plt.subplots()
plt.plot(Vla, Pla)






