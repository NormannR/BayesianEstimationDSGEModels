using JLD
using ExcelFiles
using Optim
using CsminWel

include("MCMCFunctions.jl")

funcmod = linearized_model

#Initial guess

xzero = [2.09,0.6530,2.00,0.65,0.34,3.16,0.51,1.5,4,2.5,0.19,0.65,0.5]
posest = ["tau","kappa","psi1","psi2","rA","piA","gammaQ","rho_R","rho_g","rho_z","sigma_R","sigma_g","sigma_z"]

#Data

Y = readdlm("us.txt")

#III) Transform initial guess to constrained minimization space
# println(logprior(bounds(param)))
println("Transforming initial guess to constrained minimization space")
println(logpost(xzero,funcmod,Y,1))

# IV) Minimization using CSMINWEL

funcpost = logpost_max

println("Begin Minimization")

results = optimize(x->funcpost(x,funcmod,Y,1),xzero,Csminwel(),Optim.Options(show_trace=true, iterations = 200))
xh = Optim.minimizer(results)
fh = Optim.minimum(results)
#
# save("MaxPost.jld", "xh", xh)
#V) Transform posterior from constrained minimization space to parameter
#
# var = load("MaxPost.jld")
# xh = var["xh"]

println("Transforming posterior from constrained minimization to model space")
postmode=bounds(xh)

HH = -inv(Calculus.hessian(x->funcpost(x,funcmod,Y,0), postmode))
println("Obtained Hessian!")

# Unsatisfactory Hessian : non-positive definite
# Generalized Cholesky (cf Schnabel and Eskow (1990))

L = GeneralizedCholesky(HH)
HH = L*L'
HH=0.5*(HH+HH')

data = [posest postmode sqrt.(abs.(diag(HH)))]
function Tab2Text(data,filename)
	row,col = size(data)
	open(filename, "w") do f
		for i in 1:row
			for j in 1:col
				c = data[i,j]
				write(f, "$c \t")
			end
			write(f, "\r\n")
		end
	end
end
Tab2Text(data,"tab_output.txt")

funcpost = logpost

# ========================================================================
## Begin MCMC Metropolis Algorithm
# ========================================================================
# These are extremely conservative numbers for illustration purposes only
# Should use 4 chains with ntake=50,000 and nburn = 50,000.
scaling= 0.75
nchains= 1
ntake  = 100000
nburn  = 50000

drawsMat=zeros(length(xh),ntake+nburn,nchains)

## VIII. Generate starting values for each chain
xStart      =zeros(length(xh),nchains)
logPostStart=Array{Float64,1}(nchains)
# *Note* the Hessian is scaled by c^2
#        But to generate proposal values we will use 4*(c^2)
Hscaled=scaling*scaling*HH
for ii=1:nchains
    logpostOLD=-1e10
    while (logpostOLD==-1e10) | (logpostOLD == nothing)
        xguess=rand(MvNormal(postmode,4*Hscaled))
        logpostOLD=funcpost(xguess,funcmod,Y,0)
		drawsMat[:,1,ii]=xguess
    end
    logPostStart[ii]=logpostOLD
end
println("Obtained starting values")

## IX. Random Walk Metropolis alogorithm

clockmat=Array{Float64}(nchains)
acceptmat=Array{Float64}(nchains)
for ii=1:nchains
    tic()
    println("Begin chain $ii")
    count=0
    xOLD=drawsMat[:,1,ii]
    logpostOLD=logPostStart[ii]
    for jj=2:(ntake+nburn)
        if rem(jj,10000)==0
            println("Completed draws $jj with acceptance rate $(count/jj)")
            tiempo=toc()
            clockmat[ii]=(tiempo/60) + clockmat[ii]
            println("Run time for this block $(tiempo/60)")
            tic()
        end
        # Generate Candidate
		# Generate Candidate
		logpostNEW = nothing
		xcand = nothing
		while (logpostNEW == nothing)
			xcand=rand(MvNormal(drawsMat[:,jj-1,ii],Hscaled))
			try
				logpostNEW = funcpost(xcand,funcmod,Y,0)
			catch y
				logpostNEW = nothing
			end
		end
        if logpostNEW > logpostOLD
            logpostOLD = logpostNEW
            drawsMat[:,jj,ii]=xcand
            count=count+1
        else
            if rand(1)[1]<exp(logpostNEW-logpostOLD)
                logpostOLD=logpostNEW
                drawsMat[:,jj,ii]=xcand
                count=count+1
            else
                drawsMat[:,jj,ii]=drawsMat[:,jj-1,ii]
            end
        end
    end
    acceptmat[ii]=count/(ntake+nburn)  # Acceptance Rate
end
m = mean(acceptmat)
println("Total acceptance rate $m")
println("End MCMC")

# X. Create table to summarize output

tab_sum=Array{Any}(length(xh),6)
pvec=ceil.([0.05 0.95]*ntake*nchains)
pvec=map(Int,pvec)
for ii=1:length(xh)
    xx=reshape(drawsMat[ii,nburn+1:end,:],nchains*ntake)
    xx=sort(xx)
    tab_sum[ii,1]=posest[ii]
    tab_sum[ii,2]=median(xx)
    tab_sum[ii,3]=mean(xx)
    tab_sum[ii,4]=std(xx)
    tab_sum[ii,5:6]=xx[pvec]
end

Tab2Text(tab_sum,"tab_output_est.txt")
