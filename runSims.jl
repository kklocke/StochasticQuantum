include("ising.jl")

T = parse(Float64, ARGS[1]);
n = parse(Int, ARGS[2]);
numTraj = parse(Int, ARGS[3]);
numSims = parse(Int, ARGS[4]);

print("T = ",T,", N = ", n, ", # Trajectories = ",numTraj,", # Reps = ",numSims,"\n");

if n == 10
    h = ([2.12781, -2.7702, -0.244021, -0.961017, -2.66011, 3.92394, -0.359768, -0.483442, -1.15585, 3.36586], [-3.88008, 2.5125, 2.67812, -1.31589, 2.68753, -2.71358, 2.06807, 3.62094, 3.53565, -2.82496]);
elseif n == 20
    h = ([3.59134, -3.1764, -3.94823, 3.66447, -2.37153, -2.78293, 0.32068, -2.37249, 3.293, 2.87219, 1.71039, 0.415332, 2.88892, 0.366159, -1.56891, 3.15032, 2.51979, -3.7795, 2.34391, -2.96227], [-2.53221, -3.30522, -0.562026, 2.85926, -3.46799, -3.15812, -1.88834, 0.0895059, 2.21562, -2.04881, -1.29371, -3.53577, -3.15703, -0.549962, -3.54039, 2.13074, -2.51168, 0.843927, 0.283034, -3.75319]);
elseif n == 30
    h = ([-1.95613, -2.10488, 2.18582, 3.8648, 3.39147, -2.09324, -2.68371, -2.05741, 0.500397, 1.42124, -0.193988, -3.82593, -0.901075, -0.453568, -2.21818, -0.980363, 0.442936, -0.977483, -1.84947, -1.20271, 1.14734, 3.6603, -0.401496, 3.27769, 0.718071, 1.1461, 3.694, 0.846073, 2.50228, 3.66784], [-0.0923708, -1.95757, -3.2544, -1.96441, 0.523485, 2.92897, 3.322, -0.806231, 3.94241, 3.76812, -1.45728, -3.97698, 1.73965, -0.742083, -2.69014, -0.342867, 0.27098, 3.62803, 3.4363, -3.6955, 3.28544, -3.40107, 3.04024, 2.43378, -0.0782083, 2.30896, -2.34195, 0.132948, -3.42861, 2.87535]);
elseif n == 5
    h = ([-2.00505, -0.643117, 3.26154, -0.0139472, 2.96014], [2.03901, 3.95585, -3.11311, -1.24859, -0.189046]);
else
    h = genFields(0.,4.,0.,4.,n);
end

res = zeros(Complex{Float64},numSims);
tres = T-.003;
for i = 1:numSims
    (myT,myS,mySanti,myH) = avgIsingMagnetization(1.,h,n,numTraj,1,T,.003,false,true);
    myL = length(myT);
    tres = myT[myL-1];
    @show tres
    res[i] = mean(myS[myL-1,1:n]);
end
res = real(res);
print("Mean: ", mean(res),", std: ", std(res),", T: ",tres,"\n")
print(res,"\n")
