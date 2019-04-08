using Distributions

include("exactDiag.jl")

function precomputeIsingNoise(n,J)
    precompMat = zeros(Complex{Float64},n,n);
    for m = 1:n
        k = 3.14159265*(2 * m - n)/n;
        if k > (-.5/n)
            for l = 1:n
                precompMat[l,m] = sqrt(cos(k)*J+0im)*(cos(k*l) - sin(k*l));
                if (k == 0) || (m == n)
                    precompMat[l,m] /= sqrt(2.);
                end
            end
        end
    end
    precompMat *= sqrt(2. * im/n);
    return precompMat
end

function precomputeDiag(n,J)
    Jmat = zeros(n,n) + UniformScaling(2);
    Jmat += diagm(1=>ones(n-1));
    Jmat += diagm(-1=>ones(n-1));
    Jinv = inv(Jmat);
    Jinv = Jmat;
    (D,V) = eigen(Jinv);
    res = conj(transpose(V)) * diagm(0=>sqrt.(1 ./ D));
    res = V * diagm(0=>sqrt.(D));
    res *= sqrt(J*im);
    return res;
end

function precomputeCholesky(n,J)
    Jmat = zeros(n,n) + UniformScaling(2);
    Jmat += diagm(1=>ones(n-1));
    Jmat += diagm(-1=>ones(n-1));
    # Jinv = inv(Jmat);
    # Jinv = Jmat;
    # rhoMat = zeros(n,n);
    # for i = 1:n
	# for j = 1:n
	  #   rhoMat[i,j] = Jinv[i,j]/sqrt(Jinv[i,i]*Jinv[j,j]);
          #   rhoMat[j,i] = rhoMat[i,j];
          #   if i == j
	  #       rhoMat[i,i] = 1.;
	  #   end
	# end
    # end
    # L = cholesky(rhoMat);
    # L = cholesky(Jmat);
    L = cholesky(Jmat);
    res = zeros(Complex{Float64},n,n) + convert(Array,L.L);
    res *= sqrt(J*im);
    return res;
end

function IsingUpdateUniform(J,h,n,xi,noisePre,dt=.0001,lockLast=false)
    # xi is an (n x 3) matrix containing ((xi_1+, xi_1z, xi_1-), (xi_2+ ...) ... )
    (hx,hz) = h;
    driftMat = zeros(Complex{Float64},n,3);
    for i=1:n
        driftMat[i,1] = -0.5*im*hx[i]*(1. - xi[i,1]^2)-hz[i]*im*xi[i,1];
        driftMat[i,2] = im*hx[i]*xi[i,1] - hz[i]*im;
        driftMat[i,3] = -0.5*im*hx[i]*exp(xi[i,2]);
    end

    # Make noise mass matrix
    chiMat = zeros(Complex{Float64}, n,3);
    for i=1:n
        chiMat[i,1] = 2. * xi[i,1];
        chiMat[i,2] = 2.;
        chiMat[i,3] = 0.;
    end
    chiMat *= -0.5*im;

    noiseKernels = zeros(Complex{Float64},n) + rand(Normal(0.,1.),n);
    noiseRes = zeros(Complex{Float64},n,3);
    for i=1:n
        for j=1:3
            noiseRes[i,j] = chiMat[i,j] * sum(noisePre[i,1:n] .* noiseKernels);
        end
    end
    res = xi + dt*driftMat + sqrt(dt)*noiseRes;
    if lockLast
        res[n,1:3] = zeros(Complex{Float64},3);
    end
    return res;
    # return xi + dt*driftMat + sqrt(dt)*noiseRes;
end

function IsingUpdateTransform(J,h,n,xi,noisePre,dt=0.0001)
    drift1 = zeros(Complex{Float64},n,3);
    drift2 = zeros(Complex{Float64},n,3);
    chiMat = zeros(Complex{Float64},n,3);
    noiseMat = zeros(Complex{Float64},n,3,n);
    (hx,hz) = h;

    for i=1:n
        tmpP = xi[i,1];
        tmpZ = xi[i,2];
        tmpM = xi[i,3];

        # Construct first drift term
        drift1[i,1] = 0.5*hx[i]*im*(-1 + 2*tmpP) + im*hz[i]*(1-tmpP)*tmpP;
        drift1[i,2] = -im*hx[i]*(1-tmpP)*tmpZ^2/tmpP + hz[i]*im*tmpZ^2;
        drift1[i,3] = 0.5*im*hx[i]*tmpM^2*exp((1-tmpZ)/tmpZ);

        chiMat[i,1] = -im*(1-tmpP)/tmpP;
        chiMat[i,2] = -im;
        # Multiply by precomputed part
        for a = 1:3
            for m = 1:n
                noiseMat[i,a,m] = chiMat[i,a]*noisePre[i,m];
            end
        end
    end

    # square for second drift term
    noiseMatSqr = noiseMat .* noiseMat;

    # construct drift 2
    for i = 1:n
        for a = 1:3
            drift2[i,a] = sum(noiseMatSqr[i,a,1:n])*(xi[i,a]^3);
        end
    end

    # construct new noise mat
    noiseMatTransform = zeros(Complex{Float64},n,3,n);
    for i=1:n
        for a=1:3
            for m=1:n
                noiseMatTransform[i,a,m] = -noiseMat[i,a,m]*xi[i,a]^2;
            end
        end
    end

    noiseKernels = rand(Normal(0.,1.),n);
    noiseRes = zeros(Complex{Float64},n,3);
    for i=1:n
        for j=1:3
            noiseRes[i,j] = sum(noiseMatTransform[i,j,1:n] .* noiseKernels);
        end
    end
    return xi + dt*(drift1 + drift2) + sqrt(dt)*noiseRes;
end


function simulationIsingXi(J,h,n,sampleRate=100,T=1,dt=0.00001,D=0,w=.1,lockLast=false)
    # assuming that J is just in z and h is just in x
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = zeros(Complex{Float64},n,3);
    res = zeros(Complex{Float64},1+Int(floor.(Nstep/sampleRate)[1]),n,3);
    times = zeros(Float64, 1+Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeCholesky(n,J);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            res[index,1:n,1:3] = xi;
            times[index] = dt*step;
        end
        for elem = 1:n
            if abs(real(xi[elem,1])) > 25
                return nothing
            end
        end
        htmp = (copy(h[1]),copy(h[2]));
        htmp[1][1] += D*cos(w*dt*step);
        xi = IsingUpdateUniform(J,htmp,n,xi,np,dt,lockLast);
        step += 1;
    end
    return (times, res);
end

function simulationIsingXiTransform(J,h,n,sampleRate=100,T=1,dt=0.00001)
    # assuming that J is just in z and h is just in x
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = ones(Complex{Float64},n,3);
    res = zeros(Complex{Float64},1+Int(floor.(Nstep/sampleRate)[1]),n,3);
    times = zeros(Float64, 1+Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeCholesky(n,J);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            res[index,1:n,1:3] = -1 .+ 1 ./ xi;
            times[index] = dt*step;
        end
        xi = IsingUpdateTransform(J,h,n,xi,np,dt);
        step += 1;
    end
    return (times, res);
end

function makeStates(h,n)
    hx = h[1];
    hz = h[2];
    cs = ones(n);
    for i=1:n
	if real(hx[i]) > 0.
	    cs[i] = -1.
	end
    end
    rp = zeros(Complex{Float64},n);
    rm = zeros(Complex{Float64},n);
    for i = 1:n
        rp[i] = (hx[i] + hz[i]*cs[i]) / sqrt(2*(hx[i]*hx[i] + hz[i]*hz[i]));
        rm[i] = (hx[i] - hz[i]*cs[i]) / sqrt(2*(hx[i]*hx[i] + hz[i]*hz[i]));
    end
    # rp = zeros(Complex{Float64},n);
    # rm = ones(Complex{Float64},n);
    return (-rp,rm);
    # rp = 1. / sqrt.(1.0 .+ (hz .* hz) ./ (hx .* hx));
    # rm = sqrt.(1 .- rp.*rp);
    # rm = hz .* rp ./ (abs.(hx));
    # @show rp, rm.*cs, rp .* rp .+ rm .* rm
    # return (rp, rm.*cs)
end

function simIsingU(J,h,n,T=1,dt=0.003,useRI5=true,useTransform=false)
    U = 1. + 0im;
    r = nothing;
    while (r == nothing)
        if useRI5
            r = simulationIsingXiRI5(J,h,n,1,T,dt,false,0,1);
        else
            if !useTransform
	        r = simulationIsingXi(J,h,n,1,T,dt);
            else
                r = simulationIsingXiTransform(J,h,n,1,T,dt);
            end
        end
    end
    xi = r[2][length(r[1])-1,1:n,1:3];
    for i = 1:n
        tmp = exp(-xi[i,2]/2.)*[xi[i,1]*xi[i,3] + exp(xi[i,2]) xi[i,1]; xi[i,3] 1.];
        if i == 1
	     U=tmp;
	else
	    U = kron(U,tmp);
	end
        # U = kron(U,tmp);
    end
    return (r[1][length(r[1])-1],U);
end

function avgIsingU(J,h,n,T=1,dt=0.003,numSim=100,useRI5=true,useTransform=false)
    t,U = simIsingU(J,h,n,T,dt,useRI5,useTransform);
    for j = 2:numSim
        tmp = simIsingU(J,h,n,T,dt,useRI5,useTransform);
	U += tmp[2];
    end
    U /= numSim;
    return (t,U)
end

function effH(J,h,n,dt=.001,numSim=100,useRI5=true)
    (t,U) = avgIsingU(J,h,n,2*dt,dt,numSim,useRI5);
    # myH = (ones(2^n,2^n)-U)/(dt*im)
    myH = log(U) / (-im*dt);
    return myH;
end

function simIsingMagnetization(J,h,n,sampleRate=100,T=1,dt=0.00001,useTransform=false,useRI5=false,D=0.,w=0.,lockLast=false)
    mycount = 0.;
    r1 = nothing;
    r2 = nothing;
    while (r1 == nothing)
        mycount += 1;
        if useRI5
            r1 = simulationIsingXiRI5(J,h,n,sampleRate,T,dt,useTransform,D,w,lockLast);
        else
            if useTransform
                r1 = simulationIsingXiTransform(J,h,n,sampleRate,T,dt);
            else
                r1 = simulationIsingXi(J,h,n,sampleRate,T,dt,D,w,lockLast);
            end
        end
    end
    while (r2 == nothing)
        mycount += 1;
        if useRI5
            r2 = simulationIsingXiRI5(J,h,n,sampleRate,T,dt,useTransform,D,w,lockLast);
        else
            if useTransform
                r2 = simulationIsingXiTransform(J,h,n,sampleRate,T,dt);
            else
                r2 = simulationIsingXi(J,h,n,sampleRate,T,dt,D,w,lockLast);
            end
        end
    end
    mycount -= 2;
    (t1,xi1) = r1;
    (t2,xi2) = r2;

    S = zeros(Complex{Float64},length(t1),n);
    S2 = zeros(Complex{Float64},length(t1),n);
    Santi = zeros(Complex{Float64},length(t1),n);
    Sent = ones(Complex{Float64},length(t1),n);
    rho1J = zeros(Complex{Float64},trunc(Int,length(t1)/4),n-1,4,4);
    (alpha,beta) = makeStates(h,n);
    for i=1:length(t1)
        tmpP1 = xi1[i,1:n,1];
        tmpP2 = conj(xi2[i,1:n,1]);
        tmpZ1 = xi1[i,1:n,2];
        tmpZ2 = conj(xi2[i,1:n,2]);
        tmpM1 = xi1[i,1:n,3];
        tmpM2 = conj(xi2[i,1:n,3]);
        sumZ = sum(xi1[i,1:n,2]) + conj(sum(xi2[i,1:n,2]));
        u1 = alpha.*(tmpP1.*tmpM1 .+ exp.(tmpZ1)) .+ beta.*tmpP1;
        uT1 = conj(alpha).*(tmpP2.*tmpM2 .+ exp.(tmpZ2)) .+ conj(beta).*tmpP2;
        u2 = alpha.*tmpM1 .+ beta;
        uT2 = conj(alpha).*tmpM2 .+ conj(beta);
        lambda1 = u1.*uT1 .+ u2.*uT2;
        lambda2 = 0.5*(u1.*uT1 .- u2.*uT2);
        tmpS = ones(Complex{Float64},n);
        tmpSS = ones(Complex{Float64},n);
	sumZ2 = xi1[i,1:n,2] .+ conj(xi2[i,1:n,2]);
	abcd = (u1 .* uT1) .+ (u2 .* uT2);
	for j=1:n
	    for k = j:n
		Sent[i,j] *= abcd[k]*exp(-0.5*sumZ2[k]);
	    end
	    Sent[i,j] *= log(Sent[i,j]);
	end
	for j=1:n
            for k=1:n
                if (k != j)
                    tmpS[j] *= lambda1[k]
                else
                    tmpS[j] *= lambda2[k]
                end
                if (k != j) && (k != 1)
                    tmpSS[j] *= lambda1[k];
                else
                    tmpSS[j] *= lambda2[k];
                end
            end
        end
	if (i % 4 == 0)
	    for j=2:n
		prefac = 1. + 0*im;
		for k=2:n
		    if (k != j)
			prefac *= abcd[k]*exp(-0.5*sumZ2[k]);
		    end
		end
		rho1 = exp(-0.5*sumZ2[1])*[u1[1]*uT1[1] u1[1]*uT2[1]; u2[1]*uT1[1] u2[1]*uT2[1]];
		rhoj = exp(-0.5*sumZ2[j])*[u1[j]*uT1[j] u1[j]*uT2[j]; u2[j]*uT1[j] u2[j]*uT2[j]];
		tmp1J = kron(rho1,rhoj)*prefac;
		rho1J[trunc(Int,i/4),j-1,1:4,1:4] = tmp1J;
	    end
	end
        tmpS *= exp(-0.5*sumZ);
        tmpSS *= exp(-0.5*sumZ);
        S[i,1:n] = tmpS;
        S2[i,1:n] = tmpSS;
    end
    return (t1,S,Sent,S2,rho1J,mycount);
end

function avgIsingMagnetization(J,h,n,reps,sampleRate=100,T=1,dt=0.00001,transform=false,useRI5=false,D=0.,w=0.,lockLast=false)
    skipNum = 0.;
    (t,S,Santi,S2,rho1J,c) = simIsingMagnetization(J,h,n,sampleRate,T,dt,transform,useRI5,D,w,lockLast);
    skipNum += c;
    while (isnan(S[length(t)-1]) || isnan(Santi[length(t)-1]))
        (t,S,Santi,S2,rho1J,c) = simIsingMagnetization(J,h,n,sampleRate,T,dt,transform,useRI5,D,w,lockLast);
        skipNum += 1;
	skipNum += c;
    end
    counter = reps - 1;
    while (counter > 0)
        (tmpRes, tmpRes1, tmpRes2, tmpRes3,tmpRho,c) = simIsingMagnetization(J,h,n,sampleRate,T,dt,transform,useRI5,D,w,lockLast);
        if !isnan(tmpRes1[length(t)-1]) && !isnan(tmpRes2[length(t)-1])
            S += tmpRes1;
            Santi += tmpRes2
            S2 += tmpRes3;
	    rho1J += tmpRho;
            counter -= 1;
        else
            skipNum += 1;
        end
	skipNum += c;
        if (counter % 500 == 0)
            @show counter;
        end
    end
    S /= reps;
    Santi /= reps;
    S2 /= reps;
    rho1J /= reps;
    @show skipNum;
    return (t,S,Santi,S2,rho1J);
end

function plotIsingXi(h,J,n,sampleRate=100,T=1,dt=.00005,transform=false)
    if transform
        (t,r) = simulationIsingXiTransform(J,h,n,sampleRate,T,dt);
    else
        (t,r) = simulationIsingXi(J,h,n,sampleRate,T,dt);
    end
    l = length(t)-1;
    xiP = r[1:l,1:n,1];
    xiZ = r[1:l,1:n,2];
    xiM = r[1:l,1:n,3];
    plot(t,real(xiP[1:l,1]),title="Re(Xi+)",layout=(3,2));
    for i=2:n
        plot!(t,real(xiP[1:l,i]),title="Re(Xi+)",subplot=1)
    end
    plot!(t,real(xiZ),title="Re(XiZ)",subplot=3)
    plot!(t,real(xiM),title="Re(Xi-)",subplot=5)
    plot!(t,imag(xiP),title="Im(Xi+)",subplot=2)
    plot!(t,imag(xiZ),title="Im(XiZ)",subplot=4)
    plot!(t,imag(xiM),title="Im(Xi-)",subplot=6)
    gui()
end


function computeDD(xi,n,J,h,noisePre,transform)
    tmpP = xi[1:n,1];
    tmpZ = xi[1:n,2];
    tmpM = xi[1:n,3];
    (hx,hz) = h;
    n = real(n);
    if !transform
        # drift
        driftMat = zeros(Complex{Float64},n,3);

        # driftMat[1:n,1] = 0.5*im*hx.*(tmpP.*tmpP - 1.) - im * hz .* tmpP;
        # driftMat[1:n,2] = im*hx.*tmpP - im*hz;
        # driftMat[1:n,3] = -0.5*im*hx.*exp.(tmpZ);


        for j = 1:n
            driftMat[j,1] = 0.5*im*hx[j]*(tmpP[j]^2 - 1.) - im*hz[j]*tmpP[j];
            driftMat[j,2] = im*hx[j]*tmpP[j] - im*hz[j];
            driftMat[j,3] = -0.5*im*hx[j]*exp(tmpZ[j]);
        end


        # diffusion
        chiMat = zeros(Complex{Float64},n,3);
        chiMat[1:n,1] = -im * tmpP;
        chiMat[1:n,2] = -im * ones(n);

        noiseMat = zeros(Complex{Float64},n,3,n);
	for l = 1:n
            for a = 1:3
                for m = 1:n
                    noiseMat[l,a,m] = chiMat[l,a]*noisePre[l,m];
                end
            end
        end
        return (driftMat, noiseMat);
    else
        # drift 1
        drift1 = zeros(Complex{Float64},n,3);
        drift1[1:n,1] = 0.5*im*hx.*(-1 .+ 2*tmpP) + im*hz.*(1 .- tmpP).*tmpP;
        drift1[1:n,2] = -im*hx.*(1 .- tmpP).*tmpZ.*tmpZ ./ tmpP + im*hz.*tmpZ.*tmpZ;
        drift1[1:n,3] = 0.5*im*hx.*tmpM.*tmpM .* exp.((1 .- tmpZ)./tmpZ);

        # diffusion
        chiMat = zeros(Complex{Float64},n,3);
        chiMat[1:n,1] = -im*(1 .- tmpP)./tmpP;
        chiMat[1:n,2] = -im*ones(n);

        noiseMat = zeros(Complex{Float64},n,3,n);
        for a = 1:3
            for m = 1:3
                noiseMat[1:n,a,m] = chiMat[1:n,a].*noisePre[1:n,m];
            end
        end

        noiseMatSqr = noiseMat .* noiseMat;

        drift2 = zeros(Complex{Float64},n,3);
        for i = 1:n
            for a = 1:3
                drift2[i,a] = sum(noiseMatSqr[i,a,1:n])*(xi[i,a]^3);
            end
        end

        noiseMatTransform = zeros(Complex{Float64},n,3,n);
        for i = 1:n
            for a = 1:3
                for m = 1:n
                    noiseMatTransform[i,a,m] = -noiseMat[i,a,m]*xi[i,a]^2;
                end
            end
        end
        return (drift1 + drift2, noiseMatTransform);
    end
end

Ihat = Categorical([1/6; 2/3; 1/6]);
Itilde = Bernoulli(1/2);

function genNoise(dt,num)
    tildes = sqrt(dt)*(2. * rand(Itilde, num) .- 1.);
    hatK = sqrt(3*dt)*(rand(Ihat, num) .- 2.);
    hatKL = zeros(Complex{Float64},num,num);
    for k=1:num
        for l=1:num
            if k<l
                hatKL[k,l] = .5*(hatK[k]*hatK[l] - sqrt(dt)*tildes[k]);
            elseif k > l
                hatKL[k,l] = .5*(hatK[k]*hatK[l] + sqrt(dt)*tildes[l]);
            else
                hatKL[k,l] = .5*(hatK[k]^2 - dt);
            end
        end
    end
    return (hatK, hatKL);
end

function RI5Update(xi,J,h,n,dt,noisePre,transform,lockLast=false)
    # Sample noise terms
    (noiseK, noiseKL) = genNoise(dt,n);

    # Compute some preliminary values
    aY,bY = computeDD(xi,n,J,h,noisePre,transform);
    noiseSum = bY[1:n,1:3,1] * noiseK[1];
    for k = 2:n
        noiseSum += bY[1:n,1:3,k] * noiseK[k];
    end
    # noiseSum = sum(bY .* noiseK, dims=3);
    H20 = xi + dt*aY + (1/3)*noiseSum;
    aH2,bH2 = computeDD(H20,n,J,h,noisePre,transform);
    H30 = xi + dt*(25*aY/144 + 35*aH2/144) + (-5/6)*noiseSum;
    # H30 = xi .+ dt*60*aY/144 .+ (-5/6)*sum(bY.*noiseK);

    # Compute Hk
    Hk = zeros(Complex{Float64},n,3,n,3);
    for k = 1:n
        Hk[1:n,1:3,k,1] = xi;
        Hk[1:n,1:3,k,2] = xi + .25*aY*dt + .5*bY[1:n,1:3,k]*sqrt(dt);
        Hk[1:n,1:3,k,3] = xi + .25*aY*dt - .5*bY[1:n,1:3,k]*sqrt(dt);
    end

    # Compute HhatK
    Hhatk = zeros(Complex{Float64},n,3,n,3);
    for k = 1:n
        Hhatk[1:n,1:3,k,1] = xi;
        noiseTerm = bY[1:n,1:3,1]*noiseKL[k,1];
        for j = 2:n
            noiseTerm += bY[1:n,1:3,j]*noiseKL[k,j];
        end
        # sumTerm = sum(bY.*noiseKL[k,1:n],dims=3)./sqrt(dt) .- bY[1:n,1:3,k].*noiseKL[k,k];
        sumTerm = noiseTerm/sqrt(dt) - bY[1:n,1:3,k]*noiseKL[k,k];
        Hhatk[1:n,1:3,k,2] = xi + sumTerm;
        Hhatk[1:n,1:3,k,3] = xi - sumTerm;
    end

    # Compute necessary a's and b's
    bH2k = zeros(Complex{Float64},n,3,n);
    bH3k = zeros(Complex{Float64},n,3,n);
    bHHat2k = zeros(Complex{Float64},n,3,n);
    bHHat3k = zeros(Complex{Float64},n,3,n);

    for k = 1:n
        bH2k[1:n,1:3,k] = computeDD(Hk[1:n,1:3,k,2],n,J,h,noisePre,transform)[2][1:n,1:3,k];
        bH3k[1:n,1:3,k] = computeDD(Hk[1:n,1:3,k,3],n,J,h,noisePre,transform)[2][1:n,1:3,k];
        bHHat2k[1:n,1:3,k] = computeDD(Hhatk[1:n,1:3,k,2],n,J,h,noisePre,transform)[2][1:n,1:3,k];
        bHHat3k[1:n,1:3,k] = computeDD(Hhatk[1:n,1:3,k,3],n,J,h,noisePre,transform)[2][1:n,1:3,k];
    end

    aH20 = computeDD(H20,n,J,h,noisePre,transform)[1];
    aH30 = computeDD(H30,n,J,h,noisePre,transform)[1];

    # Perform the update
    res = xi + dt*(aY/10 + 3*aH20/14 + 24*aH30/35);
    for k=1:n
        res += noiseK[k]*(bY[1:n,1:3,k] - bH2k[1:n,1:3,k] - bH3k[1:n,1:3,k]);
        res += noiseKL[k,k]*(bH2k[1:n,1:3,k] - bH3k[1:n,1:3,k])/sqrt(dt);
        res += noiseK[k]*(.5*bY[1:n,1:3,k] - .25*bHHat2k[1:n,1:3,k] - .25*bHHat3k[1:n,1:3,k]);
        res += sqrt(dt)*.5*(bHHat2k[1:n,1:3,k] - bHHat3k[1:n,1:3,k]);
    end

    if lockLast
        res[n,1:3] = zeros(Complex{Float64},3);
    end

    return res;
end

function simulationIsingXiRI5(J,h,n,sampleRate=100,T=1,dt=0.00001,transform=false,D=0.,w=0.,lockLast=false)
    # assuming that J is just in z and h is just in x
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = zeros(Complex{Float64},n,3);
    if transform
        xi = ones(Complex{Float64},n,3);# += 1.;
    end
    res = zeros(Complex{Float64},1+Int(floor.(Nstep/sampleRate)[1]),n,3);
    times = zeros(Float64, 1+Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeCholesky(n,J);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            if transform
                res[index,1:n,1:3] = -1 .+ 1 ./ xi;
            else
                res[index,1:n,1:3] = xi;
            end
	    for elem = 1:n
		if abs(real(xi[elem])) > 35
		    return nothing
		end
	    end
            times[index] = dt*step;
        end
	htmp = (copy(h[1]),copy(h[2]));
	htmp[1][1] += D * cos(w * dt * step);
        xi = RI5Update(xi,J,htmp,n,dt,np,transform,lockLast);
        step += 1;
    end
    return (times, res);
end

function genFields(hxBase,hxDisorder,hzBase,hzDisorder,n)
    hx = ones(n) * hxBase;
    hz = ones(n) * hzBase;
    if hxDisorder != 0
	hx = hx .+ rand(Uniform(-hxDisorder,hxDisorder),n);
    end
    if hzDisorder != 0
	hz = hz .+ rand(Uniform(-hzDisorder,hzDisorder),n);
    end
    return (hx,hz)
end

function simulationIsingXi_trim(J,h,n,sampleRate=100,T=1,dt=.001,D=0.,w=.1,lockLast=false)
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = zeros(Complex{Float64},n,3);
    res = zeros(Complex{Float64},1+Int(floor.(Nstep / sampleRate)[1]),n,3);
    times = zeros(Float64, 1 + Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeCholesky(n,J);

    cutIndex = -1;

    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step / sampleRate)[1])+1;
            index = max(index,1);
            res[index,1:n,1:3] = xi;
            times[index] = dt*step;
        end
        flag=false;
        for elem = 1:n
            if abs(real(xi[elem,1])) > 25
                flag=true;
                cutIndex = trunc(Int,step/sampleRate)+1;
                break;
            end
        end
        if flag
            break
        end

        htmp = (copy(h[1]),copy(h[2]));
        htmp[1][1] += D*cos(w*dt*step);
        xi = IsingUpdateUniform(J,htmp,n,xi,np,dt,lockLast);
        step += 1;
    end

    if (cutIndex > 0)
        res = res[1:max(1,cutIndex - 5),1:n,1:3];
        times = times[1:max(1,cutIndex-5)];
    end
    return (times, res);
end

function simIsingMagnetization_trim(J,h,n,sampleRate=100,T=1,dt=0.001,useRI5=false,D=0.,w=1.,lockLast=false)
    (t1,xi1) = simulationIsingXi_trim(J,h,n,sampleRate,T,dt,D,w,lockLast);
    (t2,xi2) = simulationIsingXi_trim(J,h,n,sampleRate,T,dt,D,w,lockLast);
    l = min(length(t1),length(t2));
    t = t1[1:l];
    xi1 = xi1[1:l,1:n,1:3];
    xi2 = xi2[1:l,1:n,1:3];

    S = zeros(Complex{Float64},l,n);
    S2 = zeros(Complex{Float64},l,n);
    Sent = ones(Complex{Float64},l,n);
    rho1J = zeros(Complex{Float64},trunc(Int,length(t1)/4),n-1,4,4);
    (alpha,beta) = makeStates(h,n);

    for i=1:l
        tmpP1 = xi1[i,1:n,1];
        tmpP2 = conj(xi2[i,1:n,1])
        tmpZ1 = xi1[i,1:n,2];
        tmpZ2 = conj(xi2[i,1:n,2])
        tmpM1 = xi1[i,1:n,3];
        tmpM2 = conj(xi2[i,1:n,3])

        sumZ = sum(tmpZ1) + sum(tmpZ2);
        u1 = alpha.*(tmpP1.*tmpM1 .+ exp.(tmpZ1)) .+ beta.*tmpP1;
        uT1 = conj(alpha).*(tmpP2.*tmpM2 .+ exp.(tmpZ2)) .+ conj(beta).*tmpP2;
        u2 = alpha.*tmpM1 .+ beta;
        uT2 = conj(alpha).*tmpM2.+conj(beta);
        lambda1 = u1.*uT1 .+ u2.*uT2;
        lambda2 = .5*(u1.*uT1 .- u2.*uT2);

        tmpS = ones(Complex{Float64},n);
        tmpSS = ones(Complex{Float64},n);
        sumZ2 = tmpZ1 .+ tmpZ2;
        abcd = (u1 .* uT1) .+ (u2 .* uT2);

        for j=1:n
            for k = j:n
                Sent[i,j] *= abcd[k]*exp(-0.5*sumZ2[k]);
            end
            Sent[i,j] *= log(Sent[i,j])
        end

        for j=1:n
            for k=1:n
                if (k != j)
                    tmpS[j] *= lambda1[k]
                else
                    tmpS[j] *= lambda2[k]
                end
                if (k != j) && (k != 1)
                    tmpSS[j] *= lambda1[k]
                else
                    tmpSS[j] *= lambda2[k]
                end
            end
        end

        if (i % 4 == 0)
            for j=2:n
                prefac = 1. + 0*im;
                for k=2:n
                    if (k != j)
                        prefac *= abcd[k]*exp(-0.5*sumZ2[k])
                    end
                end
                rho1 = exp(-0.5*sumZ2[1])*[u1[1]*uT1[1] u1[1]*uT2[1]; u2[1]*uT1[1] u2[1]*uT2[1]];
                rhoj = exp(-0.5*sumZ2[j])*[u1[j]*uT1[j] u1[j]*uT2[j]; u2[j]*uT1[j] u2[j]*uT2[j]];
                tmp1J = kron(rho1,rhoj)*prefac;
                rho1J[trunc(Int,i/4),j-1,1:4,1:4] = tmp1J;
            end
        end
        tmpS *= exp(-0.5*sumZ);
        tmpSS *= exp(-0.5*sumZ);
        S[i,1:n] = tmpS;
        S2[i,1:n] = tmpSS;
    end
    return (t,S,Sent,S2,rho1J)
end

function avgIsingMagnetization_trim(J,h,n,reps,sampleRate=100,T=1,dt=.001,useRI5=false,D=0.,w=.1,lockLast=false)
    Nsteps = trunc(Int,T/(dt*sampleRate)) + 1;
    t = [dt*sampleRate*i for i=0:(Nsteps-1)]
    counts = zeros(Nsteps)
    S = zeros(Complex{Float64},Nsteps,n);
    Sent = zeros(Complex{Float64},Nsteps,n);
    S2 = zeros(Complex{Float64},Nsteps,n-1);
    rho1J = zeros(Complex{Float64},trunc(Int,Nsteps/4),n-1,4,4)
    counter = reps;

    while (counter > 0)
        res = simIsingMagnetization_trim(J,h,n,sampleRate,T,dt,useRI5,D,w,lockLast);
        for i=1:length(res[1])
            counts[i] += 1;
            S[i,1:n] += res[2][i,1:n];
            Sent[i,1:n] += res[3][i,1:n];
            S2[i,1:(n-1)] += res[4][i,1:(n-1)];
        end
        counter -= 1;
    end

    for i=1:Nsteps
        S[i,1:n] /= counts[i];
        S2[i,1:(n-1)] /= counts[i];
        Sent[i,1:n] /= counts[i];
    end

    return (t,S,Sent,S2,counts);
end
