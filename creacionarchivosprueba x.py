# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:40:47 2021

@author: marco
"""

import serial  
import matplotlib.pylab as plt 
from drawnow import *

plt.ion() 
def make_fig():     
    plt.title("Lectura pin A0", fontsize=20)     
    plt.ylim(-0.1,6)     
    plt.grid(True)     
    plt.ylabel("Tension(V)")     
    plt.xlabel("Tiempo(s)")     
    plt.plot(t,volt,'o-', label= "pin A0")     
    plt.legend(loc="upper left") 

volt=[] 
t=[] 

serial_port="COM5" 
arduino_data=serial.Serial(serial_port,9600) 
 
temp_t=0 

for i in range(51):     
    try: 
        data=arduino_data.readline()         
        string_data=str(data.decode('cp437'))         
        string_data=string_data.replace("\n", "")         
        temp_volt=float(string_data) 
        t.append(temp_t)         
        volt.append(temp_volt)         
        temp_t +=0.5         
        drawnow(make_fig)         
        print(string_data)     
    except(KeyboardInterrupt,SystemExit): 
        print("chafeó la lectura") 
r=1
arduino_data.close() 

filename="datos"+str(r)+".csv" 
myFile=open(filename,'w') 
myFile.write("Tiempo(s), Voltaje(V) \n") 
for i in range(len(t)): 
    myFile.write(str(t[i])+","+str(volt[i])+"\n") 
myFile.close() 
    
    
