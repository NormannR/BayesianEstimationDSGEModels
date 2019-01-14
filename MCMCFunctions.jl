using Gensys.gensysdt
using QuantEcon.solve_discrete_lyapunov
using Distributions

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

function boundsINV(paramest)

    boundsINV = copy(paramest)

	boundsINV[2] = bound01(paramest[2])
	boundsINV[8] = bound01(paramest[8])
	boundsINV[9] = bound01(paramest[9])
	boundsINV[10] = bound01(paramest[10])

    return boundsINV
end

function bounds(paramest)

    bounds = copy(paramest)

	bounds[2] = bound01(paramest[2])
	bounds[8] = bound01(paramest[8])
	bounds[9] = bound01(paramest[9])
	bounds[10] = bound01(paramest[10])

    return bounds
end

function bound01(x)
	return exp(x)/(1+exp(x))
end

function logpost_max(x,funcmod,Y,flag_min)
	lpost = logpost(x,funcmod,Y,flag_min)
	if lpost == nothing
		return 1e10
	else
		return lpost
	end
end

function logpost(x::Array{Float64,1},funcmod,Y,flag_min)
	param=copy(x)
    if flag_min==1
        param=bounds(param)
        logpost=1e10
    else
        logpost= -1e10
    end
	# display(x)
	# Compute prior density, requires full parameter vector
	lprior,flag_ok=logprior(param)
	if flag_ok==false
        return
    end
	# display(lprior)
	# Solve model
	GG,RR,SDX,ZZ,CC,HH,eu=funcmod(param)

	if isequal(eu,[1,1])==false
	     return
	end

	T,nn=size(Y)

	# Initialize Kalman filter
	ss=size(GG,1)
	MM=RR*(SDX')
    pshat=solve_discrete_lyapunov(GG, MM*(MM'))
	shat=zeros(ss,1)
    lht=zeros(T,1)

	# Kalman filter Loop
	for ii=1:T
	    shat,pshat,lht[ii,:]=kffast(Y[ii,:],ZZ,CC,HH,shat,pshat,GG,MM)
	end
	logpost=-((T*nn*0.5)*(log(2*pi)))+(sum(lht)+lprior)
	if flag_min==1
	    logpost=-1*logpost
	end
    return logpost
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


function GeneralizedCholesky(A)
	n = size(A,1)
	tau = eps(eltype(A))^(1/3)
	tau_bar = eps(eltype(A))^(2/3)
	mu = 0.1
	phaseone = true
	gamma = maximum(abs.(diag(A)))
	L = zeros(A)
	delta = 0.
	Pprod = eye(n,n)
	deltaprev = -1e10

	j = 1
	while (j <= n) & (phaseone == true)
		# display("j : $j")
		if (maximum(diag(A)[j:end]) < tau_bar*gamma ) | (minimum(diag(A)[j:end]) < -mu*maximum(diag(A)[j:end]))
			phaseone = false
		else
			i = j-1+indmax(diag(A)[j:end])
			# display("Step one reached : $i")
			if i != j
				# display("Shuffle")
				P = eye(n,n)
				P[i,:] = zeros(n)
				P[j,:] = zeros(n)
				P[i,j] = 1
				P[j,i] = 1
				Pprod = P*Pprod
				A = P*A*P
				L = P*L*P
			end

			if j < n && minimum( diag(A)[j+1:end] - A[j+1:end,j].^2 ./ A[j,j] ) < - mu*gamma
				phaseone = false
			else
				# display("Step 2 reached !")
				L[j,j] = sqrt(A[j,j])
				L[j+1:end,j] = A[j+1:end,j]/L[j,j]
				A[j+1:end,j+1:end] -= A[j+1:end,j]*A[j+1:end,j]'/L[j,j]^2
				j += 1

			end
		end
	end

	if phaseone == false && j == n
		delta = - A[n,n] + max(tau*(- A[n,n])/(1-tau),tau_bar*gamma)
		A[n,n] += delta
		L[n,n] = sqrt(A[n,n])
	end

	if phaseone == false && j < n
		k = j-1
		g = zeros(n)
		for i in k+1:n
			g[i] = A[i,i] - sum(abs.(A[i,k+1:i-1])) - sum(abs.(A[i+1:end,i]))
		end

		for j in k+1:n-2
			i = j-1+indmax(g[j:end])
			if i != j
				# display("Shuffle")
				P = eye(n,n)
				P[i,:] = zeros(n)
				P[j,:] = zeros(n)
				P[i,j] = 1
				P[j,i] = 1
				Pprod = P*Pprod
				A = P*A*P
				L = P*L*P
			end
			normj = sum(abs.(A[j+1:end]))
			delta = max(0., - A[j,j] + max(normj, tau_bar*gamma), deltaprev)
			if delta > 0.
				A[j,j] += delta
				deltaprev = delta
			end
			if A[j,j] != normj
				temp = 1-normj/A[j,j]
				for i in j+1:n
					g[i] = g[i] + abs(A[i,j])*temp
				end
			end
			L[j,j] = sqrt(A[j,j])
			L[j+1:end,j] = A[j+1:end,j]/L[j,j]
			A[j+1:end,j+1:end] -= A[j+1:end,j]*A[j+1:end,j]'/L[j,j]^2
		end

		lambda = eigvals([ A[n-1,n-1] A[n,n-1] ; A[n,n-1] A[n,n] ])
		lambda_low = lambda[1]
		lambda_high = lambda[2]
		delta = max(0., -lambda_low + max(tau*(lambda_high - lambda_low)/(1-tau) , tau_bar*gamma), deltaprev)
		if delta > 0
			A[n-1,n-1] = A[n-1,n-1] + delta
			A[n,n] = A[n,n] + delta
			deltaprev = delta
		end
		L[n-1,n-1] = sqrt(A[n-1,n-1])
		L[n,n-1] = A[n,n-1]/L[n-1,n-1]
		L[n,n] = sqrt(A[n,n] - L[n,n-1]^2)

	end

	return (Pprod')*L*(Pprod)

end
