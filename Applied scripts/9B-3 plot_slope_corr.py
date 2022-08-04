# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:31:15 2022

@author: Sverre Hassing
"""
import numpy as np
import matplotlib.pyplot as plt

# Rough file used to create figure B-3 in the thesis

# Input values
v = 1800  # [m/s]
alpha = 5.1/180*np.pi # [rad]

# Font size
fsize=16

def find_roots(p_old, v, alpha):
    a = (np.tan(alpha))**2
    b = 0
    c = 1 - p_old**2 * (np.tan(alpha))**2
    d = 2*p_old/v * np.tan(alpha) * np.sqrt(1 - (p_old*v)**2)
    e = p_old**2 - 1/v**2

    return np.roots([a,b,c,d,e])

# Set up the range of tested slownesses. Velocity cannot be lower than medium
# velocity
min_vel = v # minimum velocity [m/s]
amt_p_vals = 2400 # amount of slowness values tested, see it as the resolution
p_range = np.linspace(-1/min_vel,1/min_vel,amt_p_vals)

# Extract all real roots from the equation
p1 = []
p2 = []
p3 = []
for p in p_range:
    roots = find_roots(p,v,alpha)
    roots = roots[np.isreal(roots)]
    p1.append(np.real(roots[roots>=0.][0]))
    p2.append(np.real(roots[np.logical_and(roots<=0.,roots>-1.)][0]))
p1 = np.array(p1)
p2 = np.array(p2)

# Plot showing the resulting real roots of the equation, so the verticaly slowness
plt.figure(dpi=300, figsize=(10,10))
plt.plot(p_range,p1)
plt.plot(p_range,p2)
plt.xlabel("Input slowness")
plt.ylabel("Vertical slowness")
plt.grid()
ax = plt.gca()
for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fsize)
plt.show()

# Plot showing the difference between the input and output slownesses
p_calc = np.sqrt(v**(-2) - p1**2)
p_calc[p_range<0] *= -1
plt.figure(dpi=300,figsize=(6,6))
plt.plot(p_range,p_calc, label = 'True slowness')
plt.plot(p_range,p_range, ls='--', label='Apparent slowness')
plt.grid()
plt.xlabel("Input slowness [s/m]")
plt.ylabel("Output slowness [s/m]")
plt.xlim([p_range[0],p_range[-1]])
plt.legend()
ax = plt.gca()
for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(0.8*fsize)
plt.show()

# Plot showing the relative error of the input slowness
p_calc = np.sqrt(v**(-2) - p1**2)
p_calc[p_range<0] *= -1
plt.figure(dpi=300,figsize=(8,8))
plt.plot(p_range,abs((p_calc-p_range)/p_calc))
# plt.plot(p_range,p_range, ls='--')
plt.yscale('log')
plt.grid()
plt.xlim([p_range[0],p_range[-1]])
plt.xlabel("Input slowness")
plt.ylabel("Relative error of input slowness compared to output slowness")
ax = plt.gca()
for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fsize)
plt.show()

# Plot showing absolute difference between input and output slowness
p_calc = np.sqrt(v**(-2) - p1**2)
p_calc[p_range<0] *= -1
plt.figure(dpi=300,figsize=(6,6))
plt.plot(p_range,p_calc-p_range)
# plt.plot(p_range,p_range, ls='--')
# plt.yscale('log')
plt.grid()
plt.xlim([p_range[0],p_range[-1]])
plt.xlabel("Input slowness [s/m]")
plt.ylabel("Difference between input and output slowness [s/m]")
ax = plt.gca()
for item in ([ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(0.6*fsize)
plt.show()

#%%

# Plot showing relative difference between the input and output velocity
v_range = 1/p_range[1200:]
v_calc = 1/p_calc[1200:]
v_diff = v_calc-v_range

plt.figure(dpi=300,figsize=(10,10))
plt.scatter(v_range,abs((v_calc-v_range)/v_range))
# plt.plot(p_range,p_range, ls='--')
plt.xscale('log')
plt.grid()
# plt.xlim([v_range[0],v_range[-1]])
plt.xlabel("Input velocity")
plt.ylabel("Difference between input and output velocity")
plt.show()