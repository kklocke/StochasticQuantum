
using LinearAlgebra
include("exactDiag.jl")
include("ising.jl")

hx = [-2.00505 -.643117 3.26154 -0.0139472];
hz = [2.03901 3.95585 -3.11311 -1.24859];
hy = [0 0 0 0];
h = (hx,hy,hz)

function eye(n)
    res = zeros(Complex{Float64},n,n);
    res += UniformScaling(n);
    return res/n;
end

k = 1. +0im;

H00 = constructH(h,(0,0,0.),4);
H01 = constructH(h,(0,0,sqrt(k)*0.01),4);
H03 = constructH(h,(0,0,sqrt(k)*0.03),4);
H10 = constructH(h,(0,0,sqrt(k)*0.10),4);

D00,V00 = eigen(H00);
D01,V01 = eigen(H01);
D03,V03 = eigen(H03);
D10,V10 = eigen(H10);

Vt00 = conj(transpose(V00));
Vt01 = conj(transpose(V01));
Vt03 = conj(transpose(V03));
Vt10 = conj(transpose(V10));

states = makeStates((hx,hz),4);
states0 = kron(kron(kron([states[1][1],states[2][1]],[states[1][2],states[2][2]]),[states[1][3],states[2][3]]),[states[1][4],states[2][4]]);
@show sum(states0 .* conj(states0))

T = 25.6
dt = 0.001
L = trunc(Int,T/dt);

states00 = zeros(Complex{Float64},L,16);
states01 = zeros(Complex{Float64},L,16);
states03 = zeros(Complex{Float64},L,16);
states10 = zeros(Complex{Float64},L,16);

for i=1:L
    t = dt*i
    U00 = V00 * diagm(0=>exp.(-im*t*D00))* Vt00;
    U01 = V01 * diagm(0=>exp.(-im*t*D01))* Vt01;
    U03 = V03 * diagm(0=>exp.(-im*t*D03))* Vt03;
    U10 = V10 * diagm(0=>exp.(-im*t*D10))* Vt10;
    states00[i,1:16] = U00 * states0;
    states01[i,1:16] = U01 * states0;
    states03[i,1:16] = U03 * states0;
    states10[i,1:16] = U10 * states0;
end

function ket(s1,s2,s3,s4)
    r1 = [1 0];
    r2 = [1 0];
    r3 = [1 0];
    r4 = [1 0];
    if s1 == 1
	r1 = [0 1];
    end
    if s2 == 1
	r2 = [0 1];
    end
    if s3 == 1
	r3 = [0 1];
    end
    if s4 == 1
	r4 = [0 1];
    end
    return kron(kron(kron(r1,r2),r3),r4);
end

function rhoElem(rho,a1,b1,c1,d1,a2,b2,c2,d2)
    myBra = ket(a1,b1,c1,d1);
    myKet = conj(transpose(ket(a2,b2,c1,d2)));
    # myKet = ket(s1,s2,s3,s4);
    # return (myKet*rho*conj(transpose(myKet)))[1,1];
    return (myBra * rho * myKet)[1,1];
end

function rhoElem1J(rho,ind,v1,v2,v3,v4)
    if ind == 2
        return (rhoElem(rho,v1,v2,0,0,v3,v4,0,0) + rhoElem(rho,v1,v2,0,1,v3,v4,0,1) + rhoElem(rho,v1,v2,1,0,v3,v4,1,0) + rhoElem(rho,v1,v2,1,1,v3,v4,1,1));
    end
    if ind == 3
	return (rhoElem(rho,v1,0,v2,0,v3,0,v4,0) + rhoElem(rho,v1,0,v2,1,v3,0,v4,1) + rhoElem(rho,v1,1,v2,0,v3,1,v4,0) + rhoElem(rho,v1,1,v2,1,v3,1,v4,1));
    end
    if ind == 4
        return (rhoElem(rho,v1,0,0,v2,v3,0,0,v4) + rhoElem(rho,v1,0,1,v2,v3,0,1,v4) + rhoElem(rho,v1,1,0,v2,v3,1,0,v4) + rhoElem(rho,v1,1,1,v2,v3,1,1,v4));
    end
end

function getSubRho(rho,ind)
    rhoSub = zeros(Complex{Float64},4,4);
    rhoSub[1,1] = rhoElem1J(rho,ind,0,0,0,0);
    rhoSub[1,2] = rhoElem1J(rho,ind,0,0,0,1);
    rhoSub[1,3] = rhoElem1J(rho,ind,0,0,1,0);
    rhoSub[1,4] = rhoElem1J(rho,ind,0,0,1,1);
    rhoSub[2,1] = rhoElem1J(rho,ind,0,1,0,0);
    rhoSub[2,2] = rhoElem1J(rho,ind,0,1,0,1);
    rhoSub[2,3] = rhoElem1J(rho,ind,0,1,1,0);
    rhoSub[2,4] = rhoElem1J(rho,ind,0,1,1,1);
    rhoSub[3,1] = rhoElem1J(rho,ind,1,0,0,0);
    rhoSub[3,2] = rhoElem1J(rho,ind,1,0,0,1);
    rhoSub[3,3] = rhoElem1J(rho,ind,1,0,1,0);
    rhoSub[3,4] = rhoElem1J(rho,ind,1,0,1,1);
    rhoSub[4,1] = rhoElem1J(rho,ind,1,1,0,0);
    rhoSub[4,2] = rhoElem1J(rho,ind,1,1,0,1);
    rhoSub[4,3] = rhoElem1J(rho,ind,1,1,1,0);
    rhoSub[4,4] = rhoElem1J(rho,ind,1,1,1,1);
    return rhoSub;
end

S2s = zeros(Complex{Float64},L,4);
S3s = zeros(Complex{Float64},L,4);
S4s = zeros(Complex{Float64},L,4);

function rho2EE(rho)
    evals = eigvals(rho) .+ 0.0im;
    logvals = log.(evals);
    sum = 0.;
    for i=1:4
        if evals[i] != 0
	    sum += evals[i]*logvals[i];
        end
    end
    return sum;
end

function rho2Ms(rho)
    S = 0.5*[1 0; 0 -1];
    myI = [1 0; 0 1];
    S1 = kron(kron(kron(S,myI),myI),myI);
    S2 = kron(kron(kron(myI,S),myI),myI);
    S3 = kron(kron(kron(myI,myI),S),myI);
    S4 = kron(kron(kron(myI,myI),myI),S);
    M1 = tr(rho*S1);
    M2 = tr(rho*S2);
    M3 = tr(rho*S3);
    M4 = tr(rho*S4);
    return (M1,M2,M3,M4);
end

function rho2Ms2(rho)
    S = [0.5 0; 0. -0.5];
    myI = [1 0; 0 1];
    S1 = kron(S,myI);
    S2 = kron(myI,S);
    SS = kron(S,S);
    return tr(rho*S2);
end

function rho2SS(rho)
    S = 0.5*[1 0; 0 -1];
    SS = kron(S,S);
    myI = [1 0; 0 1];
    S1 = kron(S,myI);
    S2 = kron(myI,S);
    SSexp = tr(rho*SS);
    S1exp = tr(rho*S1);
    S2exp = tr(rho*S2);
    return (SSexp - S1exp*S2exp);
end

function rho2SS2(rho)
    S = 0.5*[1 0; 0 -1];
    SS = kron(S,S);
    myI = [1 0; 0 1];
    S1 = kron(kron(kron(S,myI),myI),myI);
    S2 = kron(kron(kron(myI,S),myI),myI);
    S3 = kron(kron(kron(myI,myI),S),myI);
    S4 = kron(kron(kron(myI,myI),myI),S);
    S12 = kron(kron(kron(S,S),myI),myI);
    S13 = kron(kron(kron(S,myI),S),myI);
    S14 = kron(kron(kron(S,myI),myI),S);
    corr12 = tr(rho*S12)-tr(rho*S1)*tr(rho*S2);
    corr13 = tr(rho*S13)-tr(rho*S1)*tr(rho*S3);
    corr14 = tr(rho*S14)-tr(rho*S1)*tr(rho*S4);
    return (corr12, corr13, corr14);
end

SS2s = zeros(Complex{Float64},L,4);
SS3s = zeros(Complex{Float64},L,4);
SS4s = zeros(Complex{Float64},L,4);
M1s = zeros(Complex{Float64},L,4);
M2s = zeros(Complex{Float64},L,4);
M3s = zeros(Complex{Float64},L,4);
M4s = zeros(Complex{Float64},L,4);

for i=1:L
    rho00 = states00[i,1:16] * conj(transpose(states00[i,1:16]))
    rho12_00 = getSubRho(rho00,2);
    rho13_00 = getSubRho(rho00,3);
    rho14_00 = getSubRho(rho00,4);
    S2s[i,1] = rho2EE(rho12_00);
    S3s[i,1] = rho2EE(rho13_00);
    S4s[i,1] = rho2EE(rho14_00);
    (c1,c2,c3) = rho2SS2(rho00);
    SS2s[i,1] = c1; # rho2SS(rho12_00);
    SS3s[i,1] = c2; # rho2SS(rho13_00);
    SS4s[i,1] = c3; # rho2SS(rho14_00);
    (m1,m2,m3,m4) = rho2Ms(rho00);
    M1s[i,1] = m1;
    M2s[i,1] = m2;
    M3s[i,1] = m3;
    M4s[i,1] = m4;
    rho01 = states01[i,1:16]*conj(transpose(states01[i,1:16]));
    rho12_01 = getSubRho(rho01,2);
    rho13_01 = getSubRho(rho01,3);
    rho14_01 = getSubRho(rho01,4);
    S2s[i,2] = rho2EE(rho12_01);
    S3s[i,2] = rho2EE(rho13_01);
    S4s[i,2] = rho2EE(rho14_01);
    (c1,c2,c3) = rho2SS2(rho01);
    SS2s[i,2] = c1; # rho2SS(rho12_01);
    SS3s[i,2] = c2; # rho2SS(rho13_01);
    SS4s[i,2] = c3; # rho2SS(rho14_01);
    (m1,m2,m3,m4) = rho2Ms(rho01);
    M1s[i,2] = m1;
    M2s[i,2] = m2;
    M3s[i,2] = m3
    M4s[i,2] = m4;
    rho03 = states03[i,1:16]*conj(transpose(states03[i,1:16]));
    rho12_03 = getSubRho(rho03,2);
    rho13_03 = getSubRho(rho03,3);
    rho14_03 = getSubRho(rho03,4);
    S2s[i,3] = rho2EE(rho12_03);
    S3s[i,3] = rho2EE(rho13_03);
    S4s[i,3] = rho2EE(rho14_03);
    (c1,c2,c3) = rho2SS2(rho03);
    SS2s[i,3] = c1; # rho2SS(rho12_03);
    SS3s[i,3] = c2; # rho2SS(rho13_03);
    SS4s[i,3] = c3; # rho2SS(rho14_03);
    (m1,m2,m3,m4) = rho2Ms(rho03);
    M1s[i,3] = m1;
    M2s[i,3] = m2;
    M3s[i,3] = m3;
    M4s[i,3] = m4;
    rho10 = states10[i,1:16]*conj(transpose(states10[i,1:16]));
    rho12_10 = getSubRho(rho10,2);
    rho13_10 = getSubRho(rho10,3);
    rho14_10 = getSubRho(rho10,4);
    S2s[i,4] = rho2EE(rho12_10);
    S3s[i,4] = rho2EE(rho13_10);
    S4s[i,4] = rho2EE(rho14_10);
    (c1,c2,c3) = rho2SS2(rho10);
    SS2s[i,4] = c1; # rho2SS(rho12_10);
    SS3s[i,4] = c2; # rho2SS(rho13_10);
    SS4s[i,4] = c3; # rho2SS(rho14_10);
    (m1,m2,m3,m4) = rho2Ms(rho10);
    M1s[i,4] = m1;
    M2s[i,4] = m2;
    M3s[i,4] = m3;
    M4s[i,4] = m4;
end

open("N4_ED_EE.txt","w") do f
    for i=1:L
        t = dt*i;
        write(f,"$(t) $(real(S2s[i,1])) $(real(S2s[i,2])) $(real(S2s[i,3])) $(real(S2s[i,4])) ");
        write(f,"$(real(S3s[i,1])) $(real(S3s[i,2])) $(real(S3s[i,3])) $(real(S3s[i,4])) ");
        write(f,"$(real(S4s[i,1])) $(real(S4s[i,2])) $(real(S4s[i,3])) $(real(S4s[i,4]))\n");
    end
end


open("N4_ED_SS.txt","w") do f
    for i = 1:L
        t = dt * i;
        write(f,"$(t) $(real(SS2s[i,1])) $(real(SS2s[i,2])) $(real(SS2s[i,3])) $(real(SS2s[i,4])) ");
        write(f,"$(real(SS3s[i,1])) $(real(SS3s[i,2])) $(real(SS3s[i,3])) $(real(SS3s[i,4])) ");
        write(f,"$(real(SS4s[i,1])) $(real(SS4s[i,2])) $(real(SS4s[i,3])) $(real(SS4s[i,4]))\n");
    end
end

open("N4_ED_M.txt","w") do f
    for i = 1:L
        t = dt * i;
        write(f,"$(t) $(real(M1s[i,1])) $(real(M1s[i,2])) $(real(M1s[i,3])) $(real(M1s[i,4])) ");
        write(f,"$(real(M2s[i,1])) $(real(M2s[i,2])) $(real(M2s[i,3])) $(real(M2s[i,4])) ");
        write(f,"$(real(M3s[i,1])) $(real(M3s[i,2])) $(real(M3s[i,3])) $(real(M3s[i,4])) ");
        write(f,"$(real(M4s[i,1])) $(real(M4s[i,2])) $(real(M4s[i,3])) $(real(M4s[i,4]))\n");
    end
end
