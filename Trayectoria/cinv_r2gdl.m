function [q1 q2]=cinv_r2gdl(l1,l2,x0,y0)
q2=acos((x0.*x0+y0.*y0-l1*l1-l2*l2)/(2*l1*l2));
 q1=atan(y0./x0)-atan((l2*sin(q2))./(l1+l2*cos(q2)));
end