clc
clear all
close all
syms l1 l2 t1 t2 t1p t2p t r
J= [l1*cos(t1)+l2*cos(t1+t2) l2*cos(t1+t2);
    l1*sin(t1)+l2*sin(t1+t2) l2*sin(t1+t2)]
Jinv=simplify(inv(J))
Jr=[ -l1*t1p*sin(t1)-l2*(t1p+t2p)*sin(t1+t2) -l2*(t1p+t2p)*sin(t1++t2);
    l1*t1p*cos(t1)+l2*(t1p+t2p)*cos(t1+t2) l2*(t1p+t2p)*cos(t1+t2)];
Ja=[(tp1^2)(-l1*sin(t1)
t_dp=[-r*cos(t);r*sin(t)]
