f=imread('E:\Project\EvaluationCode\IRVI\IR16.png')
f=double(f);
[m,n]=size(f); 
pp=(m-1)*(n-1); 
td=0; 
for i=1:m-1 
    for j=1:n-1 
        td=td+sqrt(((f(i,j)-f(i+1,j)).^2+(f(i,j)-f(i,j+1)).^2)/2.0); 
    end 
end 
td=td/pp; 
