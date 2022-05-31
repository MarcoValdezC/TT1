function [x0, y0,z0]=cinematica_r2gdl(beta1,l1,q1,beta2,l2,q2)
dato1=whos('beta1'); dato2=whos('l1');
dato3=whos('q1');
dato4=whos('beta2'); dato5=whos('l2');
dato6=whos('q2');
v1=strcmp(dato1.class, 'sym');
v2=strcmp(dato2.class, 'sym');
v3=strcmp(dato3.class, 'sym');
v4=strcmp(dato4.class, 'sym');
v5=strcmp(dato5.class, 'sym');
v6=strcmp(dato6.class, 'sym');
digits(3);

if (v1 & v2 & v3 & v4 & v5 & v6) %caso simbólico

x0=simplify(vpa(l1*cos(q1)+l2*cos(q1+q2),3));

y0=simplify(vpa(l1*sin(q1)+l2*sin(q1+q2),3));
    z0=vpa(beta1+beta2,3);
else %caso numérico
 x0=l1*cos(q1)+l2*cos(q1+q2);
 y0=l1*sin(q1)+l2*sin(q1+q2);
 z0=beta1+beta2;
end
end