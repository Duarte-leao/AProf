clear
%%
H=4;
W=7;
M=3;
N=2;
H_prime= H - M +1;
W_prime = W - N+1;
Filter = reshape(1:M*N,[M,N]);

for i=1:H_prime*W_prime
    for j = 1:H*W
        if (mod(i-1,H_prime)<mod(j-1,H)+1) && (mod(j-1,H)+1<= mod(i-1,H_prime)+ M) && ((floor((i-1)/H_prime))*H<j)&& (j<=(floor((i-1)/H_prime)+2)*H)
            i
            j
            r(i,j)= Filter( mod(j-1,H)+1-mod(i-1,H_prime),floor((j-1)/H)+1 - floor((i-1)/H_prime));
        else
            r(i,j)=0;   
        end
    end
end