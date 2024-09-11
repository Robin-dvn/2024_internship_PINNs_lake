
using Statistics
using JLD2
"""
    loadTimesFromExp(exppath)

Charge les temps de simulation à partir d'une structure de fichiers HDF5 organisée en sous-dossiers. La fonction explore récursivement les sous-dossiers `/strat` et `/mixed` pour collecter les temps de simulation enregistrés dans les fichiers `.h5`.

# Arguments
- `exppath`: Le chemin du dossier principal contenant les sous-dossiers avec les fichiers HDF5.

# Retour
- Renvoie un tuple contenant deux listes : `strat_times` et `mixed_times` qui contiennent les temps de simulation extraits des fichiers dans les sous-dossiers `/strat` et `/mixed`, respectivement.
"""
function loadTimesFromExp(exppath)

    # Dossier principal
    main_folder = exppath
    strat_times = []
    mixed_times = []
    # Itérer sur les sous-dossiers
    for entry in readdir(main_folder)
        lambdafold = joinpath(main_folder, entry)
        if isdir(lambdafold)
            lambdastrat = lambdafold*"/strat"
            for entry in readdir(lambdastrat)
                subfolder = joinpath(lambdastrat, entry)
                
                if isdir(subfolder)
                    # Trouver le fichier dans le sous-dossier
                    files = readdir(subfolder)
                    for file in files 
                        file_path = joinpath(subfolder, file)
                        
                        if isfile(file_path) && endswith(file, ".h5")
                            # Appeler la fonction de traitement sur le fichier trouvé
                            params = loadTuple(file_path) 
                            push!(strat_times,params[:time])
                        end
                    end
                end
            end

            lambdamixed = lambdafold*"/mixed"
            # Itérer sur les sous-dossiers
            for entry in readdir(lambdamixed)
                subfolder = joinpath(lambdamixed, entry)

                if isdir(subfolder)

                    # Trouver le fichier dans le sous-dossier
                    files = readdir(subfolder)
                    for file in files 
                        file_path = joinpath(subfolder, file)
                        if isfile(file_path) && endswith(file, ".h5")
                            # Appeler la fonction de traitement sur le fichier trouvé
                            params = loadTuple(file_path) 
                            push!(mixed_times,params[:time])
                        end
                    end
                end
            end

        end
    end



    
    return (strat_times,mixed_times)
end

using Plots
using ComponentArrays
using Plots.Measures

"""
    PlotLossFromExp(exppath)

Génère et affiche des graphiques de la fonction de perte (loss) à partir des fichiers HDF5 situés dans les sous-dossiers `/strat` et `/mixed`. Deux graphiques sont produits pour les fichiers stratifiés, et un pour les fichiers mixtes.

# Arguments
- `exppath`: Le chemin du dossier principal contenant les sous-dossiers avec les fichiers HDF5.

# Retour
- Renvoie l'objet du graphique pour la partie stratifiée.
"""
function PlotLossFromExp(exppath)
    plotlyjs()
    gr()
    pltstrat = nothing
    # Dossier principal
    main_folder = exppath
    # Itérer sur les sous-dossiers
    for entry in readdir(main_folder)
        lambdafold = joinpath(main_folder, entry)
        if isdir(lambdafold)
            lambdastrat = lambdafold*"/strat"
            for entry in readdir(lambdastrat)
                subfolder = joinpath(lambdastrat, entry)
                
                if isdir(subfolder)
                    # Trouver le fichier dans le sous-dossier
                    files = readdir(subfolder)
                    for file in files 
                        file_path = joinpath(subfolder, file)
                        
                        if isfile(file_path) && endswith(file, ".h5")
                            # Appeler la fonction de traitement sur le fichier trouvé
                            params = loadTuple(file_path)
                            plt1  = plot([params[:totalLoss], params[:dataloss],params[:physloss]],label = [ "Total Loss" "Data Loss" "PhysLoss"], title = "Loss functions Straification",xlabel="Itérations",ylabel = "Loss",yaxis = :log10,figsize = (500,1500))

                            plt2  = plot([params[:totalLoss], params[:dataloss],params[:physloss]],label = [ "Total Loss" "Data Loss" "PhysLoss"], title = "Zoom",xlabel="Itérations",ylabel = "Loss",yaxis = :log10,xlims = (1,600),margin = 10mm)

                            l = @layout [a{0.5w} b]
                            pltstrat = plot(plt1,plt2,layout = l,size = (1000,400))
                            display(pltstrat)
                        end
                    end
                end
            end

            lambdamixed = lambdafold*"/mixed"
            # Itérer sur les sous-dossiers
            for entry in readdir(lambdamixed)
                subfolder = joinpath(lambdamixed, entry)

                if isdir(subfolder)

                    # Trouver le fichier dans le sous-dossier
                    files = readdir(subfolder)
                    for file in files 
                        file_path = joinpath(subfolder, file)
                        if isfile(file_path) && endswith(file, ".h5")
                            # Appeler la fonction de traitement sur le fichier trouvé
                            params = loadTuple(file_path)
                            plt = plot([params[:totalLoss], params[:dataloss],params[:physloss]],label = [ "Total Loss" "Data Loss" "PhysLoss"], title = "Loss functions mixed")
                            display(plt)
                        end
                    end
                end
            end

        end
    end


    gr()
    return pltstrat

end
"""
    PlotTestLossGraph(exppath)

Affiche des graphiques des pertes de test (`testloss`) en fonction de différents paramètres `lambda`, pour les données stratifiées et mixtes. Les pertes moyennes sont calculées pour chaque `lambda` et représentées graphiquement.

# Arguments
- `exppath`: Le chemin du dossier principal contenant les sous-dossiers avec les fichiers HDF5.

# Retour
- Renvoie le graphique comparant les pertes de test pour la stratification et le mélange.
"""
function PlotTestLossGraph(exppath)
    # plotlyjs()
    # Dossier principal
    main_folder = exppath
    # Itérer sur les sous-dossiers
    lambdas = []
    st_mean_test_loss = []
    mx_mean_test_loss = []
    for entry in readdir(main_folder)
        lambdafold = joinpath(main_folder, entry)
        # Utilisation des expressions régulières pour extraire la valeur décimale
        if isdir(lambdafold)
            m = match(r"([\d\.eE\+\-]+)", entry)

            # Vérifiez si une correspondance a été trouvée et convertissez-la en nombre
            if match !== nothing
                lambda = parse(Float64, m.match)
                push!(lambdas,lambda)
                println("La valeur décimale extraite est: $lambda")
            else
                println("Aucune valeur décimale trouvée dans le nom du dossier.")
            end
        end

        if isdir(lambdafold)
            lambdastrat = lambdafold*"/strat"
            testlosses = []
            for entry in readdir(lambdastrat)
                subfolder = joinpath(lambdastrat, entry)
                
                if isdir(subfolder)
                    # Trouver le fichier dans le sous-dossier
                    files = readdir(subfolder)
                    for file in files 
                        file_path = joinpath(subfolder, file)
                        
                        if isfile(file_path) && endswith(file, ".h5")
                            # Appeler la fonction de traitement sur le fichier trouvé
                            params = loadTuple(file_path)
                            push!(testlosses,params[:testloss])
                        end
                    end
                end
            end
            push!(st_mean_test_loss,mean(testlosses))
            testlosses = []
            lambdamixed = lambdafold*"/mixed"
            # Itérer sur les sous-dossiers
            for entry in readdir(lambdamixed)
                subfolder = joinpath(lambdamixed, entry)

                if isdir(subfolder)

                    # Trouver le fichier dans le sous-dossier
                    files = readdir(subfolder)
                    for file in files 
                        file_path = joinpath(subfolder, file)
                        if isfile(file_path) && endswith(file, ".h5")
                            # Appeler la fonction de traitement sur le fichier trouvé
                            params = loadTuple(file_path)
                            push!(testlosses,params[:testloss])
                        end
                    end
                end
            end
            push!(mx_mean_test_loss,mean(testlosses))
           

        end
    end

    st_plt = scatter(lambdas, st_mean_test_loss,color=:red,xscale = :log10,legend =false,marker=:cross,title = "Testloss Stratified",xlabel = "lambda",ylabel = "TestMSE")
    mx_plt = scatter(lambdas, mx_mean_test_loss,color=:red,xscale = :log10,legend =false,marker=:cross,title = "Testloss Mixed",xlabel = "Lambda",ylabel = "TestMSE")    
    plt = plot(st_plt,mx_plt, layout = (1,2))
    display(plt)
    gr()
    return plt 
end

plt = PlotTestLossGraph("./test_pourloss_(_T05_T15_T25_T45)_2024-08-19_19_59_34")
loadTimesFromExp("./test_pourloss_(_T05_T15_T25_T45)_2024-08-19_19_59_34")
plssave = PlotLossFromExp("./test_pourloss_(_T05_T15_T25_T45)_2024-08-19_20_26_37")
"""
    namedtuplefromfile(group)

Reconstitue un `NamedTuple` à partir d'un groupe HDF5. Si un sous-groupe est trouvé, la fonction est appelée récursivement pour reconstruire l'arborescence des données.

# Arguments
- `group`: Le groupe HDF5 ou fichier à lire.

# Retour
- Renvoie un `NamedTuple` reconstruit à partir du fichier ou du groupe HDF5.
"""

function namedtuplefromfile(group)
    dict = Dict()
   
    for name in keys(group)
        obj = group[name]
        if isa(obj,HDF5.Group)
            dict[name]=namedtuplefromfile(obj)
        elseif isa(obj,HDF5.Dataset)
            dict[name] = read(obj)
        end
    end
    return NamedTuple((Symbol(key),value) for (key,value) in dict)

end
"""
    loadTuple(path)

Charge un `NamedTuple` à partir d'un fichier HDF5.

# Arguments
- `path`: Le chemin du fichier HDF5 à lire.

# Retour
- Renvoie un `NamedTuple` reconstruit à partir des données du fichier HDF5.
"""
function loadTuple(path)
    file = h5open(path,"r") 
    tuple = namedtuplefromfile(file)
    close(file)
    return tuple
end
    
