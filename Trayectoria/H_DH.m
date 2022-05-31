function [R vect_d vect_cero c]=H_DH(H)
for i=1:3
 for j=1:3
 R(i,j)=H(i,j);
 end
end
%estructura de la matriz de transformación homogénea
vect_d=[H(1,4); H(2,4); H(3,4)];
vect_cero=[0;0;0]';
c=1;
end