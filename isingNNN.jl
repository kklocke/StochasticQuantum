include("ising.jl")

function precomputeIsingNoise_NNN(n,J1, J2)
    precompMat = zeros(Complex{Float64},n,n);
    for m = 1:n
        k = (2. * m - n)/n;
        if k >= 0
            for l = 1:n
                precompMat[l,m] = sqrt(cos(k)*J1 + cos(2*k)*J2 + 0*im)*(cos(k*l) + sin(k*l));
            end
        end
    end
    precompMat *= sqrt(2. * im/n);
    return precompMat
end

function genField(h,n)
    (hBase, hDisorder) = h;
    myField = hBase*ones(Complex{Float64},n);
    if hDisorder != 0
        myField += rand(Uniform(-hDisorder,hDisorder),n);
    end
    return myField
end

function simulationIsingXiRI5_NNN(J,h,n,sampleRate=100,T=1,dt=0.00001,transform=false)
    (J1,J2) = J;
    if isa(h,Tuple)
        h = genField(h,n)
    end
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = zeros(Complex{Float64},n,3);
    if transform
        xi += 1.;
    end
    res = zeros(Complex{Float64},1+Int(floor.(Nstep/sampleRate)[1]),n,3);
    times = zeros(Float64, 1+Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeIsingNoise_NNN(n,J1,J2);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            if transform
                res[index,1:n,1:3] = -1 + 1 ./ xi;
            else
                res[index,1:n,1:3] = xi;
            end
            times[index] = dt*step;
        end
        xi = RI5Update(xi,J1,h,n,dt,np,transform);
        step += 1;
    end
    return (times, res, h);
end

function simMagnetizationRI5_NNN(J,h,n,sampleRate=1,T=1,dt=0.003,transform=false)
    if isa(h,Tuple)
        h = genField(h,n);
    end
    (t1,xi1) = simulationIsingXiRI5_NNN(J,h,n,sampleRate,T,dt,transform);
    (t2,xi2) = simulationIsingXiRI5_NNN(J,h,n,sampleRate,T,dt,transform);
    S = zeros(Complex{Float64},length(t1),n);
    Santi = zeros(Complex{Float64},length(t1),n);
    for i=1:length(t1)
        for j=1:n
            sumPart = 0.;
            prodPart = 1.;
            prodPart2 = 1.;
            for k = 1:n
                lambdaK1 = xi1[i,k,1]*xi1[i,k,3] + exp(xi1[i,k,2]);
                lambdaK2 = conj(xi2[i,k,1])*conj(xi2[i,k,3]) + exp(conj(xi2[i,k,2]));
                sumPart += (xi1[i,k,2] + conj(xi2[i,k,2]));
                if (k != j)
                    prodPart *= (1 + xi1[i,k,1]*conj(xi2[i,k,1]));
                end
                if (k % 2 == 0)
                    if (k == j)
                        prodPart2 *= (lambdaK1*lambdaK2 - xi1[i,k,3]*conj(xi2[i,k,3]));
                    else
                        prodPart2 *= (lambdaK1*lambdaK2 + xi1[i,k,3]*conj(xi2[i,k,3]));
                    end
                else
                    if (k == j)
                        prodPart2 *= (1 - xi1[i,k,1]*conj(xi2[i,k,1]));
                    else
                        prodPart2 *= (1 + xi1[i,k,1]*conj(xi2[i,k,1]));
                    end
                end
            end
            S[i,j] = -0.5*exp(-0.5*sumPart)*prodPart*(1-xi1[i,j,1]*conj(xi2[i,j,1]));
            Santi[i,j] = -0.5*exp(-0.5*sumPart)*prodPart2;
        end
    end
    return (t1,S,Santi,h);
end

function avgMagnetizationRI5_NNN(J,h,n,reps,sampleRate=100,T=1,dt=0.00001,transform=false)
    skipNum = 0.;
    if length(h[1]) == 1
        h = genField(h,n);
    end
    (t,S,Santi) = simMagnetizationRI5_NNN(J,h,n,sampleRate,T,dt,transform);
    while (isnan(S[length(t)-1]) || isnan(Santi[length(t)-1]))
        (t,S,Santi) = simMagnetizationRI5_NNN(J,h,n,sampleRate,T,dt,transform);
        skipNum += 1;
    end
    counter = reps - 1;
    while (counter > 0)
        (tmpRes, tmpRes1, tmpRes2) = simMagnetizationRI5_NNN(J,h,n,sampleRate,T,dt,transform);
        if !isnan(tmpRes1[length(t)-1]) && !isnan(tmpRes2[length(t)-1])
            S += tmpRes1;
            Santi += tmpRes2
            counter -= 1;
        else
            skipNum += 1;
        end
        if (counter % 100 == 0)
            @show counter;
        end
    end
    S /= reps;
    Santi /= reps;
    @show skipNum;
    return (t,S,Santi,h);
end
