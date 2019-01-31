addprocs(4)
#
# #=========================================================================#
# #                           Environment
# #=========================================================================#
#
@everywhere begin
    using Gensys.gensysdt
    using QuantEcon.solve_discrete_lyapunov
    using Distributions
    using JLD

    include("SMCfunctions.jl")
end

posest = ["tau","kappa","psi1","psi2","rA","piA","gammaQ","rho_R","rho_g","rho_z","sigma_R","sigma_g","sigma_z"]

funcmod = linearized_model
@eval @everywhere funcmod = $funcmod

#Copies pour parallélisation

bound = [  0. 1e10 ;
            0. 1 ;
            1. 1e10 ;
            0. 1e10 ;
            0. 1e10 ;
            0. 1e10 ;
            0. 1e10 ;
            0. 1. ;
            0. 1. ;
            0. 1. ;
            0. 1e10 ;
            0. 1e10 ;
            0. 1e10 ]
@eval @everywhere bound=$bound

#=========================================================================#
#                           Load data
#=========================================================================#

Y = readdlm("us.txt")

@eval @everywhere Y = $Y

#=========================================================================#
#                       Set Parameters of Algorithm
#=========================================================================#

#SMC parameters

Nblock = 1
Nparam = 13
Npart = 4000
Nphi = 400
lambda = 2

#MH parameters

c = 0.5
acpt = 0.25
trgt = 0.25

#Tempering schedule

phi = ((0:(Nphi-1))/(Nphi-1)).^lambda

#Storing the results

drawsMat = zeros(Nphi, Npart, Nparam)
weightsMat   = zeros(Npart, Nphi)
constMat    = zeros(Nphi)
loglh   = zeros(Npart)
logpost = zeros(Npart)
nresamp = 0

cMat    = zeros(Nphi,1)
ESSMat  = zeros(Nphi,1)
acptMat = zeros(Nphi,1)
rsmpMat = zeros(Nphi,1)

@everywhere f(x1,x2) = Likelihoods!(x1,x2,Y,funcmod,bound)

# f([2.09, 0.657686, 2.0, 0.65, 0.34, 3.16, 0.51, 0.817574, 0.982014, 0.924142, 0.19, 0.65, 0.5],1)

#Conversion for parallelization

loglh = convert(SharedArray,loglh)
logpost = convert(SharedArray,logpost)
drawsMat = convert(SharedArray,drawsMat)
loglh_temp = SharedArray{Float64,1}(Npart)
logpost_temp = SharedArray{Float64,1}(Npart)
acpt_temp = SharedArray{Float64,1}(Npart)

#=========================================================================#
#             Initialize Algorithm: Draws from prior
#=========================================================================#

println("SMC starts....")

weightsMat[:,1] = 1/Npart
constMat[1] = sum(weightsMat[:,1])

drawsMat[1,:,:] = PriorDraws(drawsMat[1,:,:],f)

@sync @parallel for i in 1:Npart
# for i in 1:Npart
    logpost[i], loglh[i] = f(drawsMat[1,i,:],phi[1])
end


smctime   = tic()
totaltime = 0

println("SMC recursion starts...")

for i in 2:Nphi

#-----------------------------------
# (a) Correction
#-----------------------------------

    incwt = exp.((phi[i]-phi[i-1])*loglh)
    weightsMat[:, i] = weightsMat[:, i-1].*incwt
    constMat[i]     = sum(weightsMat[:, i])
    weightsMat[:, i] /= constMat[i]

    #-----------------------------------
    # (b) Selection
    #-----------------------------------

    ESS = 1/sum(weightsMat[:, i].^2)

    if (ESS < 0.5*Npart)

        id = MultinomialResampling(weightsMat[:, i])
        drawsMat[i-1, :, :] = drawsMat[i-1, id, :]

        loglh              = loglh[id]
        logpost            = logpost[id]
        weightsMat[:, i]   = 1/Npart
        nresamp            = nresamp + 1
        rsmpMat[i]         = 1

    end


    #--------------------------------------------------------
    # (c) Mutation
    #--------------------------------------------------------

    c = c*(0.95 + 0.10*exp(16*(acpt-trgt))/(1 + exp(16*(acpt-trgt))))

    para      = drawsMat[i-1, :, :]
    wght      = repmat(weightsMat[:, i], 1, Nparam)

    mu        = sum(para.*wght,1)
    z         = (para - repmat(mu, Npart, 1))
    R       = (z.*wght)'*z

    Rdiag   = diagm(diag(R))
    Rchol   = chol(Hermitian(R))'
    Rchol2  = sqrt.(Rdiag)

    tune = Array{Any,1}(4)
    tune[1] = c
    tune[2] = R
    tune[3] = Nparam
    tune[4] = phi

    loglh = convert(SharedArray,loglh)
    logpost = convert(SharedArray,logpost)

    @sync @parallel for j in 1:Npart
    # for j in 1:Npart
        ind_para, ind_loglh, ind_post, ind_acpt = MutationRWMH(para[j,:], loglh[j], logpost[j], tune, i, f)
        drawsMat[i,j,:] = ind_para
        loglh_temp[j]       = ind_loglh
        logpost_temp[j]     = ind_post
        acpt_temp[j] = ind_acpt
    end

    loglh = loglh_temp
    logpost = logpost_temp

    acpt = mean(acpt_temp)

    cMat[i,:]    = c
    ESSMat[i,:]  = ESS
    acptMat[i,:] = acpt

    if mod(i, 1) == 0

        para = drawsMat[i, :, :]
        wght = repmat(weightsMat[:, i], 1, Nparam)

        mu  = sum(para.*wght,1)

        sig = sum((para - repmat(mu, Npart, 1)).^2 .*wght,1)
        sig = (sqrt.(sig))

        totaltime = totaltime + toc()
        avgtime   = totaltime/i
        remtime   = avgtime*(Nphi-i)

        print("-----------------------------------------------\n")
        print(" Iteration = $i / $Nphi \n")
        print("-----------------------------------------------\n")
        print(" phi  = $(phi[i]) \n")
        print("-----------------------------------------------\n")
        print("  c    = $c\n")
        print("  acpt = $acpt\n")
        print("  ESS  = $ESS  ($nresamp total resamples.)\n")
        print("-----------------------------------------------\n")
        print("  time elapsed   = $totaltime\n")
        print("  time average   = $avgtime\n")
        print("  time remained  = $remtime\n")
        print("-----------------------------------------------\n")
        print("para      mean    std\n")
        print("------    ----    ----\n")
        for k in 1:Nparam
            print("$(posest[k])     $(mu[k])    $(sig[k])\n")
        end

        tic()
    end

end

print("-----------------------------------------------\n")
println("logML = $(sum(log.(constMat)))")
print("-----------------------------------------------\n")
