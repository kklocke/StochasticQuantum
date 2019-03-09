using Distributions

include("exactDiag.jl")

function precomputeIsingNoise(n,J)
    precompMat = zeros(Complex{Float64},n,n);
    for m = 1:n
        k = (2. * m - n)/n;
        if k >= 0
            for l = 1:n
                precompMat[l,m] = sqrt(cos(k)*J)*(cos(k*l) + sin(k*l));
            end
        end
    end
    precompMat *= sqrt(2. * im/n);
    return precompMat
end

function IsingUpdateUniform(J,h,n,xi,noisePre,dt=.0001)
    # xi is an (n x 3) matrix containing ((xi_1+, xi_1z, xi_1-), (xi_2+ ...) ... )

    driftMat = zeros(Complex{Float64},n,3);
    for i=1:n
        driftMat[i,1] = -0.5*im*h*(1. - xi[i,1]^2);
        driftMat[i,2] = im*h*xi[i,1];
        driftMat[i,3] = -0.5*im*h*exp(xi[i,2]);
    end

    # Make noise mass matrix
    chiMat = zeros(Complex{Float64}, n,3);
    for i=1:n
        chiMat[i,1] = 2. * xi[i,1];
        chiMat[i,2] = 2.;
        chiMat[i,3] = 0.;
    end
    chiMat *= -0.5*im;

    noiseMat = zeros(Complex{Float64}, n,3,n);
    for l = 1:n
        for a = 1:3
            for m = 1:n
                noiseMat[l,a,m] = chiMat[l,a]*noisePre[l,m];
            end
        end
    end

    noiseKernels = rand(Normal(0.,1.),n);

    noiseRes = reshape(reshape(noiseMat, n*3,n) * noiseKernels,n,3);
    return xi + dt*driftMat + sqrt(dt)*noiseRes;
end

function IsingUpdateTransform(J,h,n,xi,noisePre,dt=0.0001)
    drift1 = zeros(Complex{Float64},n,3);
    drift2 = zeros(Complex{Float64},n,3);
    chiMat = zeros(Complex{Float64},n,3);
    noiseMat = zeros(Complex{Float64},n,3,n);

    for i=1:n
        tmpP = xi[i,1];
        tmpZ = xi[i,2];
        tmpM = xi[i,3];

        # Construct first drift term
        drift1[i,1] = 0.5*h*im*(-1 + 2*tmpP);
        drift1[i,2] = -im*h*(1-tmpP)*tmpZ^2/tmpP;
        drift1[i,3] = 0.5*im*h*tmpM^2*exp((1-tmpZ)/tmpZ);

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
    noiseRes = reshape(reshape(noiseMatTransform,n*3,n)*noiseKernels, n,3);
    return xi + dt*(drift1 + drift2) + sqrt(dt)*noiseRes;
end


function simulationIsingXi(J,h,n,sampleRate=100,T=1,dt=0.00001)
    # assuming that J is just in z and h is just in x
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = zeros(Complex{Float64},n,3);
    res = zeros(Complex{Float64},1+Int(floor.(Nstep/sampleRate)[1]),n,3);
    times = zeros(Float64, 1+Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeIsingNoise(n,J);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            res[index,1:n,1:3] = xi;
            times[index] = dt*step;
        end
        xi = IsingUpdateUniform(J,h,n,xi,np,dt);
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
    np = precomputeIsingNoise(n,J);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            res[index,1:n,1:3] = -1 + 1 ./ xi;
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
    rp = (hx .+ hz .* cs) ./ sqrt.(2 .*(hx.*hx + hz.*hz));
    rm = (hx .- hz .* cs) ./ sqrt.(2 .*(hx.*hx + hz.*hz));
    return (-rp,rm);
end

function simIsingMagnetization(J,h,n,sampleRate=100,T=1,dt=0.00001,useTransform=false,useRI5=false,D=0.,w=0.)
    mycount = 0.;
    if useRI5
	r1 = nothing;
	r2 = nothing;
	# mycount = 0;
	while (r1 == nothing)
	    mycount += 1;
	    r1 = simulationIsingXiRI5(J,h,n,sampleRate,T,dt,useTransform,D,w);
        end
	while (r2 == nothing)
	    mycount += 1;
	    r2 = simulationIsingXiRI5(J,h,n,sampleRate,T,dt,useTransform,D,w);
	end
	mycount -= 2;
	(t1,xi1) = r1;
	(t2,xi2) = r2;
    else
        if useTransform
            (t1,xi1) = simulationIsingXiTransform(J,h,n,sampleRate,T,dt);
            (t2,xi2) = simulationIsingXiTransform(J,h,n,sampleRate,T,dt);
        else
            (t1,xi1) = simulationIsingXi(J,h,n,sampleRate,T,dt);
            (t2,xi2) = simulationIsingXi(J,h,n,sampleRate,T,dt);
        end
    end
    S = zeros(Complex{Float64},length(t1),n);
    S2 = zeros(Complex{Float64},length(t1),n);
    Santi = zeros(Complex{Float64},length(t1),n);
    Sent = zeros(Complex{Float64},length(t1),n);
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
	Sent[i,1:n] = exp.(-0.5 * sumZ2) .* abcd .* (log.(abcd) .- 0.5 * sumZ2);
        # Sent[i,1:n] = exp.(-0.5 * sumZ2) .* abcd;
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
        tmpS *= exp(-0.5*sumZ);
        tmpSS *= exp(-0.5*sumZ);
        S[i,1:n] = tmpS;
        S2[i,1:n] = tmpSS;
        # for j=1:n
        #     sumPart = 0.;
        #     prodPart = 1.;
        #     prodPart2 = 1.;
        #     prodPart3 = 1.;
        #     for k = 1:n
        #         lambdaK1 = xi1[i,k,1]*xi1[i,k,3] + exp(xi1[i,k,2]);
        #         lambdaK2 = conj(xi2[i,k,1])*conj(xi2[i,k,3]) + exp(conj(xi2[i,k,2]));
        #         sumPart += (xi1[i,k,2] + conj(xi2[i,k,2]));
        #         if (k != j)
        #             prodPart *= (1 + xi1[i,k,1]*conj(xi2[i,k,1]));
        #             prodPart3 *= (lambdaK1*lambdaK2 + xi1[i,k,3]*conj(xi2[i,k,3]));
        #         else
        #             prodPart *= (1 - xi1[i,k,1]*conj(xi2[i,k,1]));
        #             prodPart3 *= (lambdaK1*lambdaK2 - xi1[i,k,3]*conj(xi2[i,k,3]));
        #         end
        #         if (k % 2 == 0)
        #             if (k == j)
        #                 prodPart2 *= (lambdaK1*lambdaK2 - xi1[i,k,3]*conj(xi2[i,k,3]));
        #             else
        #                 prodPart2 *= (lambdaK1*lambdaK2 + xi1[i,k,3]*conj(xi2[i,k,3]));
        #             end
        #         else
        #             if (k == j)
        #             prodPart2 *= (1 - xi1[i,k,1]*conj(xi2[i,k,1]));
        #             else
        #                 prodPart2 *= (1 + xi1[i,k,1]*conj(xi2[i,k,1]));
        #             end
        #         end
        #     end
        #     S[i,j] = -0.5*exp(-0.5*sumPart)*prodPart;
        #     S2[i,j] = -0.5*exp(-0.5*sumPart)*prodPart3;
        #     Santi[i,j] = -0.5*exp(-0.5*sumPart)*prodPart2;
        # end
    end
    return (t1,S,Sent,S2,mycount);
end

function avgIsingMagnetization(J,h,n,reps,sampleRate=100,T=1,dt=0.00001,transform=false,useRI5=false,D=0.,w=0.)
    skipNum = 0.;
    (t,S,Santi,S2,c) = simIsingMagnetization(J,h,n,sampleRate,T,dt,transform,useRI5,D,w);
    skipNum += c;
    while (isnan(S[length(t)-1]) || isnan(Santi[length(t)-1]))
        (t,S,Santi,S2,c) = simIsingMagnetization(J,h,n,sampleRate,T,dt,transform,useRI5,D,w);
        skipNum += 1;
	skipNum += c;
    end
    counter = reps - 1;
    while (counter > 0)
        (tmpRes, tmpRes1, tmpRes2, tmpRes3,c) = simIsingMagnetization(J,h,n,sampleRate,T,dt,transform,useRI5,D,w);
        if !isnan(tmpRes1[length(t)-1]) && !isnan(tmpRes2[length(t)-1])
            S += tmpRes1;
            Santi += tmpRes2
            S2 += tmpRes3;
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
    @show skipNum;
    return (t,S,Santi,S2);
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

        driftMat[1:n,1] = -0.5*im*hx.*(1. .- tmpP.*tmpP) .- im * hz .* tmpP;
        driftMat[1:n,2] = im*hx.*tmpP .- im*hz;
        driftMat[1:n,3] = -0.5*im*hx.*exp.(tmpZ);

        # diffusion
        chiMat = zeros(Complex{Float64},n,3);
        chiMat[1:n,1] = -im.*tmpP;
        chiMat[1:n,2] = -im.*ones(n);

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
        drift1[1:n,1] = 0.5*im*h.*(-1 + 2*tmpP);
        drift1[1:n,2] = -im*h.*(1-tmpP).*tmpZ.*tmpZ ./ tmpP;
        drift1[1:n,3] = 0.5*im*h.*tmpM.*tmpM .* exp.((1-tmpZ)./tmpZ);

        # diffusion
        chiMat = zeros(Complex{Float64},n,3);
        chiMat[1:n,1] = -im*(1-tmpP)./tmpP;
        chiMat[1:n,2] = -im;

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

function RI5Update(xi,J,h,n,dt,noisePre,transform)
    # Sample noise terms
    (noiseK, noiseKL) = genNoise(dt,n);

    # Compute some preliminary values
    aY,bY = computeDD(xi,n,J,h,noisePre,transform);
    H20 = xi .+ dt*aY .+ (1/3)*sum(bY.*noiseK);
    H30 = xi .+ dt*58*aY/144 .+ (-5/6)*sum(bY.*noiseK);

    # Compute Hk
    Hk = zeros(Complex{Float64},n,3,n,3);
    for k = 1:n
        Hk[1:n,1:3,k,1] = xi;
        Hk[1:n,1:3,k,2] = xi .+ .25*aY*dt .+ .5*bY[1:n,1:3,k]*sqrt(dt);
        Hk[1:n,1:3,k,3] = xi .+ .25*aY*dt .- .5*bY[1:n,1:3,k]*sqrt(dt);
    end

    # Compute HhatK
    Hhatk = zeros(Complex{Float64},n,3,n,3);
    for k = 1:n
        Hhatk[1:n,1:3,k,1] = xi;
        sumTerm = sum(bY.*noiseKL[k,1:n])./sqrt(dt) .- bY[1:n,1:3,k].*noiseKL[k,k];
        Hhatk[1:n,1:3,k,2] = xi .+ sumTerm;
        Hhatk[1:n,1:3,k,3] = xi .- sumTerm;
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

    return res;
end

function simulationIsingXiRI5(J,h,n,sampleRate=100,T=1,dt=0.00001,transform=false,D=0.,w=0.)
    # assuming that J is just in z and h is just in x
    Nstep = floor.(T/dt)[1];
    step = 0;
    xi = zeros(Complex{Float64},n,3);
    if transform
        xi += 1.;
    end
    res = zeros(Complex{Float64},1+Int(floor.(Nstep/sampleRate)[1]),n,3);
    times = zeros(Float64, 1+Int(floor.(Nstep / sampleRate)[1]));
    np = precomputeIsingNoise(n,J);
    while (step < Nstep)
        if (step % sampleRate == 0)
            index = Int(floor.(step/sampleRate)[1])+1;
            index = max(index,1);
            if transform
                res[index,1:n,1:3] = -1 + 1 ./ xi;
            else
                res[index,1:n,1:3] = xi;
            end
	    for elem = 1:n
		if abs(real(xi[elem])) > 20
		    return nothing
		end
	    end
            times[index] = dt*step;
        end
	htmp = h;
	htmp[1][1] += D * cos(w * dt * step);
        xi = RI5Update(xi,J,htmp,n,dt,np,transform);
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
