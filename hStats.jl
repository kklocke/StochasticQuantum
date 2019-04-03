using LinearAlgebra
include("ising.jl")
include("exactDiag.jl")

function eye(n)
    return zeros(n,n) + UniformScaling(1)
end

T = parse(Float64,ARGS[1])
J = parse(Float64, ARGS[2])
numAvg = parse(Int64,ARGS[3])
numSim = parse(Int64,ARGS[4])
dt = parse(Float64, ARGS[5])
n = parse(Int64,ARGS[6])


ht = constructH((4*ones(n),zeros(n),4*ones(n)),(0,0,J),n);

function hstats(ht,t,numavg,numsim,useRI5,dt)
    dat = zeros(numSim);
    for i = 1:numSim
        ueff = avgIsingU(J,(4*ones(n),4*ones(n)),n,t+dt,dt,numavg,useRI5,false);
        h = log(ueff[2])/(-im*ueff[1]);
        dat[i] = norm(h-ht)/norm(ht);
    end
    return (mean(dat),std(dat));
end

@show T,n,dt,J,numAvg,numSim,hstats(ht,T,numAvg,numSim,true,dt);
