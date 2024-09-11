using Pkg
cd(@__DIR__)
Pkg.activate(".")
using Distributed
# addprocs(27)
@everywhere begin 
    using Dates
    using JLD2
    using Statistics
    using DataStructures
    using Lux,NeuralPDE
    using Random
    using ComponentArrays
end

@everywhere include("PINN_utils.jl")

@everywhere function show_progress(completed, total)
    percentage = completed / total
    completed_width = Int(round(percentage * 100))
    remaining_width = 100 - completed_width

    bar = "[" * "="^completed_width * " "^(remaining_width) * "]"
    println("Progress: $bar $((percentage*100))%")
end

dir = 

function main(nom_exp,lambdas,indices_profondeurs,nb_res_st,nb_res_mx,nbiter_adam,nbiter_lbfgs, path = @__DIR__)
    
    gr()
    
    indices_a_exclure = setdiff(1:5,indices_profondeurs)
    
    df_clean = CSV.read(path*"/df_clean.csv", DataFrame) 


    T05 = df_clean[!,:Tw_05]
    T15 = df_clean[!,:Tw_15]
    T25 = df_clean[!,:Tw_25]
    T35 = df_clean[!,:Tw_35]
    T45 = df_clean[!,:Tw_45]
    temps_strings = ["T05" "T15" "T25" "T35" "T45" ]
    temps_strings_train = temps_strings[indices_profondeurs]
    temps_strings_test = temps_strings[indices_a_exclure]
    tempstringpath = ""
    for temp in temps_strings_train
        tempstringpath *= "_$temp"
    end
    templosstitle = ""
    for temp in temps_strings_test
        templosstitle *= " $temp"
    end
    temperatures = [T05,T15,T25,T35,T45]
    temperatures = temperatures[indices_profondeurs]
    profondeurs = [0.5 1.5 2.5 3.5 4.5]
    profondeurstrain = profondeurs[indices_profondeurs]
    dates = df_clean[!,:date]


    dir = @__DIR__
    date = now()
    datetime_str = Dates.format(date, "yyyy-mm-dd_HH_MM_SS")
    dirpath = dir*"//$(nom_exp)_($tempstringpath)_"*datetime_str
    mkdir(dirpath)


    inner = 25
    nbpoints_exp = 100
    nbiteradam = nbiter_adam
    nbiterlbfgs = nbiter_lbfgs

    
    configpath = dirpath*"//config.json"
    createConfigFile(inner,4,"tanh",nbpoints_exp,indices_profondeurs,nbiteradam,nbiterlbfgs,configpath)

    chain = Lux.Chain(Dense(2, inner, Lux.tanh),
                        Dense(inner, inner, Lux.tanh),
                        Dense(inner, inner, Lux.tanh),
                        Dense(inner, inner, Lux.tanh),
                        Dense(inner, 1))

    ps = Lux.setup(Random.default_rng(), chain)[1]

    pdesys = getSys()
    for i in 1:length(lambdas)
        lambda = lambdas[i]
        ## création des dossier
        lambdapath = dirpath*"//lambdaphys_$lambda"
        mkdir(lambdapath)
        stratpath = lambdapath*"//strat"
        mixedpath = lambdapath*"//mixed"
        mkdir(stratpath)
        mkdir(mixedpath)

        ## génération des différentes partitions
        st, mx = getStratMixedSpans(100)
        st_dates,mx_dates = getDateSpans(st,dates),getDateSpans(mx,dates)
        stratloss, mixedloss = getStratMixedLosses(temperatures,profondeurstrain,st,mx)
        stratdiscrs = []
        mixeddiscrs = []
        stpinreps = []
        mxpinreps = []
        stdecomposedlosses = []
        mxdecomposedlosses = []

        for (stloss,mxloss) in zip(stratloss,mixedloss)

            naloss = NonAdaptiveLoss(;pde_loss_weights = lambda, bc_loss_weights = 0., additional_loss_weights = 1.)
            # graloss = GradientScaleAdaptiveLoss(100;
            #             weight_change_inertia = 0.9,
            #             pde_loss_weights = lambda,
            #             bc_loss_weights = 0.,
            #             additional_loss_weights = 1.0)
            strategy = QuasiRandomTraining(1000;bcs_points = 1) #inutile puisque pas de poids sur les loss²
            st_dis = PhysicsInformedNN(chain,strategy;init_params = ps,adaptive_loss=naloss,additional_loss=stloss)
            mx_dis = PhysicsInformedNN(chain,strategy;init_params = ps,adaptive_loss=naloss,additional_loss=mxloss)
            stpinnrep = symbolic_discretize(pdesys,st_dis)
            strealloss = getPersonalLossFunction(stpinnrep,[lambda,0.,1.])
            mxpinnrep = symbolic_discretize(pdesys,mx_dis)
            mxrealloss = getPersonalLossFunction(mxpinnrep,[lambda,0.,1.])
            push!(stratdiscrs,st_dis)
            push!(mixeddiscrs,mx_dis)
            push!(stpinreps,stpinnrep)
            push!(mxpinreps,mxpinnrep)
            push!(stdecomposedlosses,strealloss)
            push!(mxdecomposedlosses,mxrealloss)
        end
        st_realseries = getRealSeries([T05,T15,T25,T35,T45],st)
        mx_realseries = getRealSeries([T05,T15,T25,T35,T45],mx)

        

        ## entrainement stratification
 

        for i in 1:nb_res_st
            dis,pinrep,decomposedloss,realserie,datespan  = stratdiscrs[i],stpinreps[i],stdecomposedlosses[i],st_realseries[i],st_dates[i]
            realserieloss = realserie[indices_a_exclure]
            profondeur_test = profondeurs[indices_a_exclure]
            startdate,stopdate = datespan[1], datespan[end]
            datestr = "//"*Dates.format(startdate, "yyyy-mm-dd")*"---"*Dates.format(stopdate, "yyyy-mm-dd")
            partitionpath = stratpath*datestr 
            mkdir(partitionpath)

  
            totalLoss = []
            physloss = []
            dataloss = []
            bcloss = []
            iteration = 1
            callback = function callback(p,l,args...)
                bc_loss_arr,pde_loss_arr,add_loss_arr = args
                push!(totalLoss,l)
                push!(physloss,lambda*pde_loss_arr[1])
                push!(bcloss,0*bc_loss_arr[1])
                push!(dataloss,1.0*add_loss_arr[1])
                
                iteration  = iteration +1
                if iteration % 500 == 0
                    print("Strat lambda: $lambda , n°$(floor(i))  ")
                    show_progress(iteration,21 * nbiter_adam + 2*nbiter_lbfgs)
                end
                
                return false    
            end
            time = 0
            if lambda == 0.
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = 1, nbiterlbgs = 100,adamlambda = 0.01)
                phi,θ = network
            else
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = nbiteradam, nbiterlbgs = nbiterlbfgs,adamlambda = 0.01)
                phi,θ = network
            end
            
            if length(indices_profondeurs) == 5
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,0,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (θ = θ,testloss = -1,totalLoss = totalLoss, bcloss = bcloss, dataloss = dataloss,physloss = physloss,time = time)

            else
                testloss = getTestLoss((phi,θ),realserieloss,profondeur_test)
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,testloss,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (θ = θ,testloss = testloss,totalLoss = totalLoss, bcloss = bcloss, dataloss = dataloss,physloss = physloss,time = time)
            end
            @save partitionpath*"//params.jld2" params



        end

        ## entrainement mixed layers
      

        for i in 1:nb_res_mx 
            dis,pinrep,decomposedloss,realserie,datespan  = mixeddiscrs[i],mxpinreps[i],mxdecomposedlosses[i],mx_realseries[i],mx_dates[i]
            realserieloss = realserie[indices_a_exclure]
            profondeur_test = profondeurs[indices_a_exclure]
            startdate,stopdate = datespan[1], datespan[end]
            datestr = "//"*Dates.format(startdate, "yyyy-mm-dd")*"---"*Dates.format(stopdate, "yyyy-mm-dd")
            partitionpath = mixedpath*datestr 
            mkdir(partitionpath)
            
            totalLoss = []
            physloss = []
            dataloss = []
            bcloss = []
            iteration = 1
            callback = function callback(p,l,args...)
                bc_loss_arr,pde_loss_arr,add_loss_arr = args
                push!(totalLoss,l)
                push!(physloss,lambda*pde_loss_arr[1])
                push!(bcloss,0.0*bc_loss_arr[1])
                push!(dataloss,1.0*add_loss_arr[1])
                # println("Current loss is : $l")
                
                iteration  = iteration +1
                if iteration % 500 == 0
                    print("Mixed lambda: $lambda , n°$(floor(i))  ")
                    show_progress(iteration,21 * nbiter_adam + nbiter_lbfgs)
                end
                return false    
            end
            time = 0
            if lambda == 0.
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = 1, nbiterlbgs = 100,adamlambda = 0.01)
                phi,θ = network
            else
                network,time =@timed trainPINN(dis,pinrep,decomposedloss,callback;nbiteradam = nbiteradam, nbiterlbgs = nbiterlbfgs,adamlambda = 0.01)
                phi,θ = network
            end

            if length(indices_profondeurs) == 5
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,0,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (θ = θ,testloss = -1,totalLoss = totalLoss, bcloss = bcloss, dataloss = dataloss,physloss = physloss,time = time)
            else
                testloss = getTestLoss((phi,θ),realserieloss,profondeur_test)
                pltloss = plot([totalLoss,physloss,bcloss,dataloss],label = ["total" "phy" "bc" "dataloss"])
                plotPINN((phi,θ),realserie,datespan,testloss,lambda,pltloss;anim = true,save=true,path = partitionpath)
                params = (θ = θ,testloss = testloss,totalLoss = totalLoss, bcloss = bcloss, dataloss = dataloss,physloss = physloss,time = time)
            end
            @save partitionpath*"//params.jld2" params


        end


    end


    

end

nom_exp = "Expérience2jours_ADAM0.01"
lambdas = [0.001]
#lambdas = [0.0001,0.0005,0.5,0.1,0.005,0.001,0.05,0.01]
indices_profondeurs = [1,2,3,5]
nb_res_st = 1
nb_res_mx= 1
nbiter_adam = 30
nbiter_lbfgs = 1

if nprocs()>4
    main(nom_exp,lambdas,indices_profondeurs,nb_res_st,nb_res_mx,nbiter_adam,nbiter_lbfgs)
end

# include("PINN_analysis.jl")
# PlotLossFromExp("C://Users//davenner//Documents//PINNs//PGNN-PINN-Lakes//PINNs//Apprentissage//DifferentialEquations//Expérience2jours_ADAM0.01_(_T05_T15_T25_T45)_2024-07-29_11_17_18")
# t1,t2 = loadTimesFromExp("C://Users//davenner//Documents//PINNs//PGNN-PINN-Lakes//PINNs//Apprentissage//DifferentialEquations//Expérience2jours_ADAM0.01_(_T05_T15_T25_T45)_2024-07-29_11_17_18")


# show_progress(10,100)
