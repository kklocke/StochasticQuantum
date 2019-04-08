include("ising.jl")
using LinearAlgebra
# using Plots
# plotly()

dt = 0.003;
T = parse(Float64, ARGS[1]);
n = parse(Int, ARGS[2]);
numTraj = parse(Int, ARGS[3]);
datFile = ARGS[4];

D = 0.
w = 0.
if length(ARGS) > 4
    D = parse(Float64,ARGS[5]);
    w = parse(Float64,ARGS[6]);
end

hd = 4.;
if length(ARGS) > 6
    hd = parse(Float64,ARGS[7]);
end

if n == 10
    h = ([2.12781, -2.7702, -0.244021, -0.961017, -2.66011, 3.92394, -0.359768, -0.483442, -1.15585, 3.36586], [-3.88008, 2.5125, 2.67812, -1.31589, 2.68753, -2.71358, 2.06807, 3.62094, 3.53565, -2.82496]);
    # h = ([2.12781, -2.7702, -0.244021, -0.961017, -2.66011, 3.92394, -0.359768, -0.483442, -1.15585, 3.36586], [-4.,-4.,-4.,-4.,-4.,-4.,-4.,-4.,-4.,-4.]);
elseif n == 11
    h = ([2.12781, -2.7702, -0.244021, -0.961017, -2.66011, 3.92394, -0.359768, -0.483442, -1.15585, 3.36586,0.0],[-3.88008, 2.5125, 2.67812, -1.31589,2.68753, -2.71358, 2.06807, 3.62094, 3.53565, -2.82496,-4]);
elseif n == 20
    h = ([3.59134, -3.1764, -3.94823, 3.66447, -2.37153, -2.78293, 0.32068, -2.37249, 3.293, 2.87219, 1.71039, 0.415332, 2.88892, 0.366159, -1.56891, 3.15032, 2.51979, -3.7795, 2.34391, -2.96227], [-2.53221, -3.30522, -0.562026, 2.85926, -3.46799, -3.15812, -1.88834, 0.0895059, 2.21562, -2.04881, -1.29371, -3.53577, -3.15703, -0.549962, -3.54039, 2.13074, -2.51168, 0.843927, 0.283034, -3.75319]);
elseif n == 30
    h = ([-1.95613, -2.10488, 2.18582, 3.8648, 3.39147, -2.09324, -2.68371, -2.05741, 0.500397, 1.42124, -0.193988, -3.82593, -0.901075, -0.453568, -2.21818, -0.980363, 0.442936, -0.977483, -1.84947, -1.20271, 1.14734, 3.6603, -0.401496, 3.27769, 0.718071, 1.1461, 3.694, 0.846073, 2.50228, 3.66784], [-0.0923708, -1.95757, -3.2544, -1.96441, 0.523485, 2.92897, 3.322, -0.806231, 3.94241, 3.76812, -1.45728, -3.97698, 1.73965, -0.742083, -2.69014, -0.342867, 0.27098, 3.62803, 3.4363, -3.6955, 3.28544, -3.40107, 3.04024, 2.43378, -0.0782083, 2.30896, -2.34195, 0.132948, -3.42861, 2.87535]);
elseif n == 5
    # h = ([-2.00505, -0.643117, 3.26154, -0.0139472, 2.96014], [2.03901, 3.95585, -3.11311, -1.24859, -0.189046]);
    # h = ([-3.39041, -0.276199, 1.65118, -0.283292, 0.0468881], [1.28597, 1.16333, -1.7593, -3.82061, -0.0995429]);
    h = ([2.19831, -2.85963, 2.98159, 2.42895, -3.2253], [-1.56679, 3.34093, -1.8971, 1.9457, 0.680429]);
elseif n == 4
    # h = ([4., 4., 4., 4.], [0.,0.,0.,0.,]);
    # h = ([4., 4., 4., 4.],[4., 4., 4., 4.]);
    # h = genFields(0.,hd,0.,hd,n);
    h = ([-2.00505, -.643117, 3.26154, -.0139472],[2.03901, 3.95585, -3.11311, -1.24859]);
    h = ([-2.00505, -.643117, 3.26154, -.0139472],[1, 1, 1, 1]);
    # h = ([2.480115, 3.7747296, 3.529954, -2.138984],[2.974651, 3.83645064, -1.840723, 0.59917]);
else
    h = genFields(0.,4.,0.,4.,n);
end

@show h

if length(ARGS) > 6
    dt = parse(Float64,ARGS[7]);
end

J = 1.;
if length(ARGS) > 7
    J = parse(Float64,ARGS[8]);
end

J *= -1.;

# (myT,myS,mySent,mySS,myRho) = avgIsingMagnetization(J,h,n,numTraj,1,T,dt,false,true,D,w,false);
 
(myT, myS, mySent, mySS) = avgIsingMagnetization_trim(J,h,n,numTraj,1,T,dt,false,D,w,false)

mySent2 = zeros(Complex{Float64},length(myT),n-1);
# for i=1:size(myRho)[1]
#     for j = 1:n-1
#	tmpRho = myRho[i,j,1:4,1:4] .+ 0im;
#	evals = eigvals(tmpRho) .+ 0im;
#	logVals = log.(evals);
#	for k=1:4
#	    if evals[k] == 0
#		logVals[k] = 0
#	    end
#	end
#	# Maybe just check if logvals is divergent
#	ss = sum(evals .* logVals);
#	# @show evals .* logVals 
#       mySent2[(4*i)-3,j] = ss;
#	mySent2[(4*i)-2,j] = ss;
#	mySent2[(4*i)-1,j] = ss;
#	mySent2[4*i,j] = ss;
#    end
#end


print("Compute mean\n")
res = zeros(Complex{Float64},length(myT));
for i = 1:length(myT)
    res[i] = mean(myS[i,1:n])
end
intRes = zeros(Complex{Float64},length(myT));
print("Compute integrated mean\n")
intRes[1] = res[1];
for i = 2:length(myT)
    intRes[i] = intRes[i-1] + res[i];
end
for i = 1:length(myT)
    intRes[i] /= i
end

open(datFile,"w") do f
    for i=1:length(myT)
	write(f,"$(myT[i]) $(real(res[i])) $(real(intRes[i])) ")
	for j=1:(n-1)
	    tmp = real(mySS[i,j]) - real(myS[i,1])*real(myS[i,j]);
	    # write(f,"$(real(mySS[i,j])) ")
	    write(f,"$(tmp) ")
	end
	# write(f,"\n");
	for j=1:n
	    tmp = real(myS[i,j]);
	    write(f,"$(tmp) ");
	end
	#for j=1:n-1
	#    tmp = real(mySent2[i,j]);
	#    write(f,"$(tmp) ");
	#end
	write(f,"\n")
    end
end

# print("Plot\n")
# plot(myT,real(res),xaxis=("Time"),yaxis=("Mean Magnetization"),layout=(3,1))
# plot!(myT,real(intRes),xaxis=("Time"),yaxis=("Int Mean Magnetization"),subplot=2)
# for i=1:n
#     plot!(myT,real(mySS[1:length(myT),i]),xaxis="Time",yaxis=("Correlator"),label=i,subplot=3)
# end
# 
# png(imName)
# close()
