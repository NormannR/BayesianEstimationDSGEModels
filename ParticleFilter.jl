addprocs(4)
@everywhere begin

    using Gensys.gensysdt
    using QuantEcon.solve_discrete_lyapunov
    using Distributions
    using JLD

    include("ParticleFunctions.jl")
end

posest = ["tau","kappa","psi1","psi2","rA","piA","gammaQ","rho_R","rho_g","rho_z","sigma_R","sigma_g","sigma_z"]
funcmod = linearized_model

#=========================================================================#
#                           Load data
#=========================================================================#

Y = readdlm("us.txt")

#=========================================================================#
#                           Load estimated parameters
#=========================================================================#

param = [2.09,0.98,2.25,0.65,0.34,3.16,0.51,0.81,0.98,0.93,0.19,0.65,0.24]

#=========================================================================#
#                           Kalman filter
#=========================================================================#

GG,RR,SDX,ZZ,CC,HH,eu=funcmod(param)

T =size(Y,1)
n = size(GG,1)
nobs = size(ZZ,1)
Liki = zeros(T)
MeasurePredi = zeros(T,nobs)
StatePredi = zeros(T+1,n)
VarStatePredi = zeros(T+1,n,n)

# Initialize Kalman filter

MM=RR*SDX'
VarStatePredi[1,:,:]=solve_discrete_lyapunov(GG, MM*MM')

# Kalman filter Loop

for t in 1:T
	Liki[t],MeasurePredi[t,:],StatePredi[t+1,:],VarStatePredi[t+1,:,:]=kffast(Y[t,:],ZZ,CC,HH,StatePredi[t,:],VarStatePredi[t,:,:],GG,MM)
end

LogLh = sum(Liki)

#=========================================================================#
#                           Bootstrap particle filter
#=========================================================================#

# 1. Initialization

N = 10000
Nstate = size(GG,1)
Nshock = size(RR,2)
Nobs = size(Y,2)
T = size(Y,1)

BSESS = zeros(T)
BSLiki = ones(T)
BSStates = zeros(T,Nstate,N)

s0 = zeros(Nstate)
P0 = nearestSPD(solve_discrete_lyapunov(GG, MM*MM'))

BSWeights = ones(N)
ups   = repmat(s0, 1, N) + chol(Hermitian(P0))'*randn(Nstate, N)

# tic()
# MultinomialResampling(abs.(ups[1,:]))
# toc()

println("Bootstrap Particle Filter")

for t in 1:T
    println("Period $t / $T")

    #2. (a) Forecasting

    fors = GG*ups + MM*randn(Nshock,N)

    #2. (b) Forecasting

    PredError  = repmat(Y[t,:], 1, N) - repmat(CC, 1, N) - ZZ*fors
    density = pdf(MvNormal(zeros(Nobs), HH),PredError)

    #2. (c) Updating

    NormWeights = BSWeights.*density/mean(BSWeights.*density)

    #2. (d) Selection

    BSESS[t] = N^2/sum(NormWeights.^2)
    println("ESS : $(BSESS[t])")

    if BSESS[t] >= N/2
        ups = fors
        BSWeights = NormWeights
    else
        println("Resampling is necessary")
        id = MultinomialResampling(NormWeights/sum(NormWeights))
        ups = fors[:,id]
        BSWeights = ones(N)
    end

    BSLiki[t] = log(mean(BSWeights.*density))
    BSStates[t,:,:] = ups

end

BSLogLh =sum(BSLiki)

#=========================================================================#
#                        Conditionally Optimal filter
#=========================================================================#

N = 400
Nstate = size(GG,1)
Nshock = size(RR,2)
Nobs = size(Y,2)
T = size(Y,1)

COESS = zeros(T)
COLiki = ones(T)
COStates = zeros(T,Nstate,N)
COWeights = ones(N)

s0 = zeros(Nstate)
P0 = nearestSPD(solve_discrete_lyapunov(GG, MM*MM'))

ups   = repmat(s0, 1, N) + chol(Hermitian(P0))'*randn(Nstate, N)
upP   = P0

println("CO Particle Filter")
for t in 1:T
    println("Period $t / $T")

    #2. (a) Forecasting

    #Steps from a Kalman Filter p. 186 in Herbst and Schorfeide's book

    fors = GG*ups
	forP = nearestSPD(MM*MM')
	fory = repmat(CC, 1, N)+ZZ*fors
    v  = repmat(Y[t,:], 1, N) - fory
    F = ZZ*forP*ZZ' + HH

	C = cholfact(Hermitian(F))
	z = C[:L]\v
    x = C[:U]\z
	M = forP*ZZ'
	sqrtinvF = inv(C[:L])
	invF = sqrtinvF'*sqrtinvF
	ups = fors + M*x
	upP = forP - M*invF*M'
	upP = nearestSPD(upP)

    #Draws from the CO importance sampler

    imps = ups +  chol(Hermitian(upP))'*randn(Nstate, N)

    omega = SharedArray{Float64,1}(N)
    #Computation of weights
    @sync @parallel for n in 1:N
        omega[n] = pdf(MvNormal(fors[:,n], forP),imps[:,n])./pdf(MvNormal(ups[:,n], upP),imps[:,n])
    end

    # (b) Forecasting

    PredError = repmat(Y[t,:], 1, N) - repmat(CC, 1, N) - ZZ*imps
    weights = pdf(MvNormal(zeros(Nobs), HH),PredError).*omega

    #2. (c) Updating

    NormWeights = COWeights.*weights/mean(COWeights.*weights)
	# NormWeights = weights/mean(weights)

    #2. (d) Selection

    COESS[t] = N^2/sum(NormWeights.^2)
    println("ESS : $(COESS[t])")

    if COESS[t] >= N/2
        ups = imps
        COWeights = NormWeights
    else
        println("Resampling is necessary")
        id = MultinomialResampling(NormWeights/sum(NormWeights))
        ups = imps[:,id]
        COWeights = ones(N)
    end

    COLiki[t] = log(mean(COWeights.*weights))
    COStates[t,:,:] = ups

end

COLogLh =sum(COLiki)

#=========================================================================#
#              Conditionally Optimal filter with Resample-Move
#=========================================================================#

#Parameters for tuning the MH step

NMH = 100
c = 0.5
trgt = 0.25
acpt = 0.25

#Parameters for the PF

N = 400
Nstate = size(GG,1)
Nshock = size(RR,2)
Nobs = size(Y,2)
T = size(Y,1)

COESS_RM = zeros(T)
COLiki_RM = ones(T)
COStates_RM = zeros(T,Nstate,N)
COWeights_RM = ones(N)

s0 = zeros(Nstate)
P0 = nearestSPD(solve_discrete_lyapunov(GG, RR*SDX*SDX*RR'))

ups   = repmat(s0, 1, N) + chol(Hermitian(P0))'*randn(Nstate, N)
upP   = P0
println("CO PF with RM step")
for t in 1:T
    println("Period $t / $T")

    #2. (a) Forecasting

    #Steps from a Kalman Filter p. 186 in Herbst and Schorfeide's book

    sm = ups

    fors = GG*ups
	forP = RR*SDX*SDX'*RR'
	fory = ZZ*fors
    v  = repmat(Y[t,:] - CC, 1, N) - ZZ*fors
    F = ZZ*forP*ZZ' + HH

	C = cholfact(Hermitian(F))
	z = C[:L]\v
    x = C[:U]\z
	M = forP*ZZ'
	sqrtinvF = inv(C[:L])
	invF = sqrtinvF'*sqrtinvF
	ups = fors + M*x
	upP = forP - M*invF*M'
    upP = nearestSPD(upP)
    forP = nearestSPD(forP)

    #Draws from the CO importance sampler

    imps = ups + chol(upP)'*randn(Nstate,N)
    omega = SharedArray{Float64,1}(N)

    #Computation of weights

    @sync @parallel for n in 1:N
        omega[n] = pdf(MvNormal(fors[:,n], forP),imps[:,n])./pdf(MvNormal(ups[:,n], upP),imps[:,n])
    end

    # (b) Forecasting

    PredError = repmat(Y[t,:] - CC, 1, N) - ZZ*imps
    weights = pdf(MvNormal(zeros(Nobs), HH),PredError).*omega

    #2. (c) Updating

    NormWeights = COWeights_RM.*weights/mean(COWeights_RM.*weights)

    #2. (d') Selection and Resample-Move

    #Selection

    COESS_RM[t] = N^2/sum(NormWeights.^2)
    println("ESS : $(COESS_RM[t])")

    id = MultinomialResampling(NormWeights/sum(NormWeights))
    shat = imps[:,id]
    smhat = sm[:,id]
    COWeights_RM = ones(N)

    #Resample-Move

    R = upP

    c = c*(0.95 + 0.10*exp(16*(acpt-trgt))/(1 + exp(16*(acpt-trgt))))

    sMH = copy(shat)
    temp_acpt = zeros(N)
    sMH = convert(SharedArray,sMH)
    temp_acpt = convert(SharedArray,temp_acpt)
    acptArray = zeros(NMH)

    for n in 1:NMH

        @sync @parallel for j in 1:N
            dzeta = shat[:,j] + c*chol(Hermitian(R))'*randn(Nstate)
            PredError = Y[t,:] - CC - ZZ*dzeta
            Numerator = pdf(MvNormal(zeros(Nobs),HH),PredError)
            Numerator *= pdf(MvNormal(GG*smhat[:,j],forP),dzeta)
            Numerator /= pdf(MvNormal(shat[:,j],c*c*R),dzeta)
            PredError = Y[t,:] - CC - ZZ*shat[:,j]
            Denominator = pdf(MvNormal(zeros(Nobs),HH),PredError)
            Denominator *= pdf(MvNormal(GG*smhat[:,j],forP),shat[:,j])
            Denominator /= pdf(MvNormal(dzeta,c*c*R),shat[:,j])

            alphalim = min(1,Numerator/Denominator)

            u = rand()

            if u<alphalim
                sMH[:,j] = dzeta
                temp_acpt[j] = 1
            else
                sMH[:,j] = shat[:,j]
            end

        end

        smhat = shat
        acptArray[n] = mean(temp_acpt)
        shat = copy(sMH)

    end

    acpt = mean(acptArray)

    ups = copy(shat)
    println("Scale parameter c : $c")
    println("Acceptance rate : $(acpt)")
    COLiki_RM[t] = log(mean(COWeights_RM.*weights))
    COStates_RM[t,:,:] = ups

end

COLogLh_RM =sum(COLiki_RM)

#=========================================================================#
#                  Log Likelihood Increments
#=========================================================================#
using PyPlot
figure()
plot(Liki,"blue")
plot(BSLiki,"red")
plot(COLiki,"green")
plot(COLiki_RM,"black")
title("\$ln \\left[\\hat{p} \\left(y_{t}|Y_{1:t-1}, \\Theta^{m}\\right)\\right] \$ vs. \$ln \\left[ p\\left(y_{t}|Y_{1:t-1}, \\Theta^{m}\\right)\\right] \$")
legend(["Kalman Filter","Bootstrap PF","Conditionally optimal PF","CO PF with Resample-Move"])
tight_layout()
savefig("LikIncrements.pdf")

IndShocks = [3,5,6]
Var = ["R","g","z"]
FileVar = ["R","g","z"]
for s in 1:length(IndShocks)
    figure()
    plot(StatePredi[2:end,IndShocks[s]],"blue")
    plot(BSStates[:,IndShocks[s]],"red")
    plot(COStates[:,IndShocks[s]],"green")
    plot(COStates_RM[:,IndShocks[s]],"black")
    title("\$\\hat{E} \\left(\\hat{$(Var[s])}_{t}|Y_{1:t-1}, \\Theta^{m}\\right)\$ vs. \$E\\left(\\hat{$(Var[s])}_t|Y_{1:t-1}, \\Theta^{m}\\right)\$")
    legend(["Kalman Filter","Bootstrap PF","Conditionally optimal PF","CO PF with Resample-Move"])
    tight_layout()
    savefig("ComparisonFilters_$(FileVar[s]).pdf")
end
