function linearized_model(param)

    NNY = 6
    NX = 3
    NETA = 2
    NY = NNY + NETA

    GAM0 = zeros(NY,NY)
    GAM1 = zeros(NY,NY)
    PSI = zeros(NY,NX)
    PPI = zeros(NY,NETA)
    C = zeros(NY)

	tau = param[1]
	kappa = param[2]
	psi_1 = param[3]
	psi_2 = param[4]
	r_A = param[5]
	pi_A = param[6]
	gamma_Q = param[7]
	rho_R = param[8]
	rho_g = param[9]
	rho_z = param[10]
	sig_R = param[11]
	sig_g = param[12]
	sig_z = param[13]

	beta = 1/(1+r_A/400)

    #Endogenous variables

    y = 1
    pi = 2
    R = 3
    L_y = 4
	g = 5
    z = 6
    E_y = 7
    E_pi = 8

    #Perturbations

    e_z = 1
    e_g = 2
    e_R = 3

    #Expectation errors

    eta_y = 1
    eta_pi = 2

    #Euler equation

    GAM0[1,y] = 1
    GAM0[1,E_y] = -1
    GAM0[1,R] = 1/tau
    GAM0[1,E_pi] = -1/tau
    GAM0[1,z] = -rho_z/tau
    GAM0[1,g] = -(1-rho_g)

    #NK PC

    GAM0[2,pi] = 1
    GAM0[2,E_pi] = - beta
    GAM0[2,y] = - kappa
    GAM0[2,g] = kappa

	# kappa = tau*(1-nu)/(nu*ss[pi]^2*phi)

    #Taylor rule

    GAM0[3,R] = 1
	GAM1[3,R] = rho_R
    GAM0[3,pi] = -(1-rho_R)*psi_1
    GAM0[3,y] = -(1-rho_R)*psi_2
    GAM0[3,g] = (1-rho_R)*psi_2
    PSI[3,e_R] = 1

    #Shock processes

    GAM0[4,z] = 1
    GAM1[4,z] = rho_z
    PSI[4,e_z] = 1

    GAM0[5,g] = 1
    GAM1[5,g] = rho_g
    PSI[5,e_g] = 1

    #Expectation errors

    GAM0[6,y] = 1
    GAM1[6,E_y] = 1
    PPI[6,eta_y] = 1

    GAM0[7,pi] = 1
    GAM1[7,E_pi] = 1
    PPI[7,eta_pi] = 1

	#Lagged Y

	GAM0[8,L_y] = 1
	GAM1[8,y] = 1

    #Sims

    GG, CC, RR, _, _, _, _, eu, _ = gensysdt(GAM0, GAM1, C, PSI, PPI)

    #Standard Deviations

    stdev = [sig_z,sig_g,sig_R]
    SDX = diagm(stdev)

    #Observables

    Nobs = 3
    ZZ = zeros(Nobs,NY)
	CC = zeros(Nobs)
	HH = zeros(Nobs,Nobs)
    #Aggregate

	CC[1] = gamma_Q
	CC[2] = pi_A
	CC[3] = pi_A + r_A + 4*gamma_Q

    ZZ[1,y] = 1
	ZZ[1,L_y] = -1
	ZZ[1,z] = 1


    ZZ[2,pi] = 4
    ZZ[3,R] = 4

	HH[1, y] = (0.20*0.579923)^2
	HH[2, pi] = (0.20*1.470832)^2
	HH[3, R] = (0.20*2.237937)^2

    return (GG,RR,SDX,ZZ,CC,HH,eu,NY,NNY,NETA,NX)

end

function PriorDraws(priorMat::Array{Float64,2},f)

    Npart = size(priorMat,1)
    priorMat = convert(SharedArray{Float64,2},priorMat)

    @sync @parallel for n in 1:Npart
        valid = false
        while !valid

			param1 = [2, 1.5, 0.5, 0.5, 7]
			param2 = [0.5, 0.25, 0.25, 0.5, 2]

			theta = param2.^2. ./ param1
			alpha = param1 ./ theta

            priorMat[n,1] = rand(Gamma(alpha[1],theta[1]))
			priorMat[n,3] = rand(Gamma(alpha[2],theta[2]))
			priorMat[n,4] = rand(Gamma(alpha[3],theta[3]))
			priorMat[n,5] = rand(Gamma(alpha[4],theta[4]))
			priorMat[n,6] = rand(Gamma(alpha[5],theta[5]))

            priorMat[n,[2,8,9,10]] = rand(Uniform(0,1),4)
			priorMat[n,7] = rand(Normal(0.4,0.2))
			priorMat[n,11] = rand(InverseGamma(0.4^2,4))
			priorMat[n,12] = rand(InverseGamma(1.^2,4))
			priorMat[n,13] = rand(InverseGamma(0.5^2,4))

			try
                f(priorMat[n,:],0)
                valid = true
            end

        end


    end

    return priorMat

end

function Likelihoods!(x,phi,Y,funcmod,bound)

    param = copy(x)
    outbound = (param .< bound[:,1]) .| (param .> bound[:,2])

    if any(outbound)

        lpost = -Inf
        lY = -Inf
        lprior = -Inf

    else

        GG,RR,SDX,ZZ,CC,HH,eu=funcmod(param)
    	if eu[2]!=1

             lY = - Inf

        else

            # Initialize Kalman filter

            T,nn=size(Y)
        	ss=size(GG,1)
        	MM=RR*(SDX')
            pshat=solve_discrete_lyapunov(GG, MM*(MM'))
        	shat=zeros(ss,1)
            lht=zeros(T)

        	# Kalman filter Loop

        	for ii=1:T
        	    shat,pshat,lht[ii,:]=kffast(Y[ii,:],ZZ,CC,HH,shat,pshat,GG,MM)
        	end

            lY = -((T*nn*0.5)*(log(2*pi))) + sum(lht)
            lprior,_=logprior(param)
            lpost = phi*lY + lprior

        end

    end

    return (lpost,lY,lprior)

end

function MultinomialResampling(W)

    N = length(W)
    U = rand(N)
    CumW = [sum(W[1:i]) for i in 1:N]
    A = SharedArray{Int64,1}(N)

    @sync @parallel for i in 1:N
	# for i in 1:N
        A[i] = findfirst(x->(U[i]<x),CumW)
    end

    return A

end

function MutationRWMH(p0, l0, post0, tune, i, f)

    c = tune[1]
    R = tune[2]
    Nparam = tune[3]
    phi = tune[4]

    valid = false
    px = nothing
    postx = nothing
    lx = nothing
    while !valid
        px = p0 + c*chol(Hermitian(R))'*randn(Nparam,1)
        try
            postx, lx, _ = f(px, phi[i])
            valid = true
        end
    end

    post0 = post0+(phi[i]-phi[i-1])*l0

    alp = exp(postx - post0)
    u = rand()

    if u < alp
        ind_para   = px
        ind_loglh  = lx
        ind_post   = postx
        ind_acpt   = 1
    else
        ind_para   = p0
        ind_loglh  = l0
        ind_post   = post0
        ind_acpt   = 0
    end

    return (ind_para, ind_loglh, ind_post, ind_acpt)

end

function logprior(paramest)
    prior = Array{Float64,1}(length(paramest))

	#Gamma priors

	param1 = [2, 1.5, 0.5, 0.5, 7]
	param2 = [0.5, 0.25, 0.25, 0.5, 2]

	theta = param2.^2. ./ param1
	alpha = param1 ./ theta

    prior[1] = logpdf(Gamma(alpha[1],theta[1]),paramest[1])
    prior[3] = logpdf(Gamma(alpha[2],theta[2]),paramest[3])
	prior[4] = logpdf(Gamma(alpha[3],theta[3]),paramest[4])
	prior[5] = logpdf(Gamma(alpha[4],theta[4]),paramest[5])
	prior[6] = logpdf(Gamma(alpha[5],theta[5]),paramest[6])

	#Uniform Priors

	prior[2] = logpdf(Uniform(0,1),paramest[2])
	prior[8] = logpdf(Uniform(0,1),paramest[8])
	prior[9] = logpdf(Uniform(0,1),paramest[9])
	prior[10] = logpdf(Uniform(0,1),paramest[10])

	#Normal priors

    prior[7] = logpdf(Normal(0.4,0.2),paramest[7])

	#Inverse Gamma

    prior[11] = lnpdfig(paramest[11],0.40,4)
    prior[12] = lnpdfig(paramest[12],1.00,4)
    prior[13] = lnpdfig(paramest[13],0.50,4)

    if any(isnan,prior) | any(isinf,prior)
        flag_ok=0
        lprior=NaN
    else
        flag_ok=1
        lprior=sum(prior)
    end

    return (lprior,flag_ok)

end

function lnpdfig(x,a,b)
# % LNPDFIG(X,A,B)
# %	calculates log INVGAMMA(A,B) at X
#
# % 03/03/2002
# % Sungbae An
	return log(2) - log(gamma(b/2)) + (b/2)*log(b*a^2/2) - ( (b+1)/2 )*log(x^2) - b*a^2/(2*x^2)
end

function kffast(y,Z,CC,HH,s,P,T,R)

	#Forecasting

	fors = T*s
	forP = T*P*T'+R*R'
	fory = CC + Z*fors
	forV = Z*forP*Z' + HH

	#Updating

	C = cholfact(Hermitian(forV))
	z = C[:L]\(y-fory)
	x = C[:U]\z
	M = forP'*Z'
	sqrtinvforV = inv(C[:L])
	invforV = sqrtinvforV'*sqrtinvforV
	ups = fors + M*x
	upP = forP - M*invforV*M'

	#log-Likelihood

	lh=-.5*(y-fory)'*invforV*(y-fory)-sum(log.(diag(C[:L])))
	return (ups,upP,lh,fory)
end
