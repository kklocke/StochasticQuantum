function constructH(h,J,n,pbc=false)
    H = zeros((2^n, 2^n));
    if length(h[1]) == 1
        # hx = h[1]*ones(n);
        # hy = h[2]*ones(n);
        # hz = h[3]*ones(n);
        (hx,hy,hz) = h;
    else
        (hx,hy,hz) = h;
    end
    (Jx,Jy,Jz) = J;
    Sx = [0. .5; .5 0.];
    Sy = [0. -.5*im; .5*im 0.];
    Sz = [.5 0.; 0. -.5];
    Sxx = kron(Sx,Sx);
    Syy = kron(Sy,Sy);
    Szz = kron(Sz,Sz);
    for i=1:n
        left = eye(2^(i-1))
        rightH = eye(2^(n-i));
        rightJ = eye(2^(max(0,n-i-1)));
        H += hx[i]*kron(kron(left, Sx), rightH);
        H += hy[i]*kron(kron(left, Sy), rightH);
        H += hz[i]*kron(kron(left, Sz), rightH);
        if i != n
            H += Jx*kron(kron(left,Sxx),rightJ);
            H += Jy*kron(kron(left,Syy),rightJ);
            H += Jz*kron(kron(left,Szz),rightJ);
        else
            mid = eye(2^(n-2));
	    if pbc
                H += Jx*kron(kron(Sx,mid),Sx);
                H += Jy*kron(kron(Sy,mid),Sy);
                H += Jz*kron(kron(Sz,mid),Sz);
	    end
        end
    end
    H += UniformScaling(.25*n*Jz);
    return H
end

function MagOperators(n)
    sz = [.5 0.;0. -.5];
    meanMag = zeros(Complex{Float64},(2^n,2^n));
    imbalance = zeros(Complex{Float64},(2^n,2^n));
    for i=1:n
        left = eye(2^(i-1));
        right = eye(2^(n-i));
        tmp = kron(kron(left,sz),right);
        meanMag += tmp;
        imbalance += ((-1.)^i)*tmp;
    end
    meanMag /= n;
    imbalance /= n;
    return (meanMag,imbalance);
end

function testED(h,J,n)
    H = constructH(h,J,n);
    D,V = eig(H);
    D = diagm(D);
    Vt = conj(transpose(V));
    tryH = V * D * Vt;
    diff = vecnorm(tryH-H,1);
    return (abs(diff) < 1e-10)
end

function antiferroInd(n)
    myStr = "1"
    for i = 2:n
        if isodd(i)
            myStr = string(myStr, "1")
        else
            myStr = string(myStr, "0")
        end
    end
    return 2^n - parse(Int,myStr,2);
end

function EDU(h,J,n,T)
    H = constructH(h,J,n);
    D,V = eigen(H);
    Vt = conj(transpose(V));
    U = V * diagm(0=>exp.(-im*T*D)) * Vt;
    return U;
end

function simulateED(h,J,n,T=1,dt=0.0001)
    H = constructH(h,J,n);
    D,V = eigen(H);
    Vt = conj(transpose(V));
    nsteps = Int(floor.(T/dt)[1]);
    t = zeros(nsteps);
    resM = zeros(Complex{Float64},nsteps);
    resI = zeros(Complex{Float64},nsteps);
    integratedM = zeros(Complex{Float64},nsteps);
    integratedI = zeros(Complex{Float64},nsteps);
    resMaf = zeros(Complex{Float64},nsteps);
    resIaf = zeros(Complex{Float64},nsteps);
    integratedMaf = zeros(Complex{Float64},nsteps);
    integratedIaf = zeros(Complex{Float64},nsteps);
    # afInd = antiferroInd(n);
    (M,I) = MagOperators(n);
    for i=1:nsteps
        U = V * diagm(0=>exp.(-im*dt*i*D)) * Vt;
        Ut = conj(transpose(U));
        t[i] = i*dt;
        tmpM = Ut*M*U;
        tmpI = Ut*I*U;
        resM[i] = tmpM[2^n,2^n];
        resI[i] = tmpI[2^n,2^n];
        # resMaf[i] = tmpM[afInd,afInd];
        # resIaf[i] = tmpI[afInd,afInd];
        if i == 1
            integratedM[i] = resM[i];
            integratedI[i] = resI[i];
            integratedMaf[i] = resMaf[i];
            integratedIaf[i] = resIaf[i];
        else
            integratedM[i] = integratedM[i-1] + resM[i];
            integratedI[i] = integratedI[i-1] + resI[i];
            integratedMaf[i] = integratedMaf[i-1] + resMaf[i];
            integratedIaf[i] = integratedIaf[i-1] + resIaf[i];
        end
    end
    for i=1:nsteps
        integratedM[i] /= i;
        integratedI[i] /= i;
        integratedMaf[i] /= i;
        integratedIaf[i] /= i;
    end
    return (t,resM,integratedM);
end

function constructH_NNN(h,J,n)
    (J1,J2) = J
    H = zeros((2^n,2^n));
    Sx = [0. .5; .5 0.];
    Sz = [.5 0.; 0. -.5];
    Sxx = kron(Sx,Sx);
    Szz = kron(Sz,Sz);
    for i=1:n
        left = eye(2^(i-1));
        rightH = eye(2^(n-i));
        rightJ = eye(2^(max(0,n-i-1)));
        H += h[i]*kron(kron(left, Sx), rightH);
        if i != n
            H += J1*kron(kron(left,Szz),rightJ);
        else
            mid = eye(2^(n-2));
            H += J1*kron(kron(Sz,mid),Sz);
        end
        nnn_ind = i + 2;
        if (nnn_ind > n)
            nnn_ind -= n;
        end
        left = eye(2^(max(0, min(nnn_ind, i)-1)));
        right = eye(2^(max(0,n-max(nnn_ind,i))));
        H += J2*kron(kron(left, Sz),kron(kron(eye(2),Sz),right));
    end
    return H
end


function simulateED_NNN(h,J,n,T=1,dt=0.005)
    H = constructH_NNN(h,J,n);
    D,V = eig(H);
    Vt = conj(transpose(V));
    nsteps = Int(floor.(T/dt)[1]);
    t = zeros(nsteps);
    resM = zeros(Complex{Float64},nsteps);
    resI = zeros(Complex{Float64},nsteps);
    integratedM = zeros(Complex{Float64},nsteps);
    integratedI = zeros(Complex{Float64},nsteps);
    resMaf = zeros(Complex{Float64},nsteps);
    resIaf = zeros(Complex{Float64},nsteps);
    integratedMaf = zeros(Complex{Float64},nsteps);
    integratedIaf = zeros(Complex{Float64},nsteps);
    afInd = antiferroInd(n);
    (M,I) = MagOperators(n);
    for i=1:nsteps
        U = V * diagm(exp.(-im*dt*i*D)) * Vt;
        t[i] = i*dt;
        tmpM = conj(transpose(U))*M*U;
        tmpI = conj(transpose(U))*I*U;
        resM[i] = tmpM[2^n,2^n];
        resI[i] = tmpI[2^n,2^n];
        resMaf[i] = tmpM[afInd,afInd];
        resIaf[i] = tmpI[afInd,afInd];
        if i == 1
            integratedM[i] = resM[i];
            integratedI[i] = resI[i];
            integratedMaf[i] = resMaf[i];
            integratedIaf[i] = resIaf[i];
        else
            integratedM[i] = integratedM[i-1] + resM[i];
            integratedI[i] = integratedI[i-1] + resI[i];
            integratedMaf[i] = integratedMaf[i-1] + resMaf[i];
            integratedIaf[i] = integratedIaf[i-1] + resIaf[i];
        end
    end
    for i=1:nsteps
        integratedM[i] /= i;
        integratedI[i] /= i;
        integratedMaf[i] /= i;
        integratedIaf[i] /= i;
    end
    return (t, resM, integratedM, resMaf, integratedMaf, resI, integratedI, resIaf, integratedIaf)
end
