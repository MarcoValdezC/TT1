clc
clear all
close all
syms l1 l2 t1 t2 
J= [l1*cos(t1)+l2*cos(t1+t2) l2*cos(t1+t2);
    l1*sin(t1)+l2*sin(t1+t2) l2*sin(t1+t2)]
Jinv=simplify(inv(J))