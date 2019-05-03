module RunOrderOptim

using DataFrames, CSV, Convex, GLPKMathProgInterface, Query, Tables

const _example_classes=(@__DIR__) * "\\..\\Examples\\ClassList.csv"
const _example_entries=(@__DIR__) *" \\..\\Examples\\EntriesExample.csv"

# greetings()=(@__DIR__) * "\\..\\Examples\\ClassList.csv"

function read_classes(fullpath::AbstractString=_example_classes;
    truestrings=["TRUE","True","true"],falsestrings=["FALSE","False","false"],kwargs...)

    df=CSV.read(fullpath;allowmissing=:auto,truestrings=truestrings,falsestrings=falsestrings,kwargs...)
    df=df |> @filter(_.Enable==true) |> DataFrame
    deletecols!(df,:Enable)
    df=vcat(df, DataFrame(Class="N".*('A':'Z'),ClassGroup="Novice",BumpClass=missing,))
    return df
end

function read_entries(fullpath::AbstractString=_example_entries;
    novice="N", truestrings=["TRUE","True","true"],falsestrings=["FALSE","False","false"],kwargs...)

    df=CSV.read(fullpath;allowmissing=:all,truestrings=truestrings,falsestrings=falsestrings,kwargs...)
    rename!(df, [Symbol("Last Name")=>:LastName, Symbol("First Name")=>:FirstName,
        Symbol("Modifier/PAX")=>:Index])
    df=df[[:LastName,:FirstName,:Class,:Index,:Exempt]]
    df.Novice = df |> @map(_.Class == novice) |> collect
    df.Exempt = coalesce.(df.Exempt,false)
    df = df |> @mutate(Class = _.Novice ? "N"*first(_.LastName[]) : _.Class) |> Tables.rows |> DataFrame
    df.IndexedClass = coalesce.(df.Index,df.Class)

    return df
end

function setup_problem(classes,entries;run_groups::Integer=2,keep_empty=false,
    separate_classes=Vector{Pair{String,String}}[],
    min_class_size::Integer=4, workerweight=10, driverweight=2, noviceweight=1)
    @assert workerweight > 0
    @assert driverweight > 0
    @assert noviceweight > 0

    # @info separate_classes
    df=_class_counts(classes,entries;keep_empty=keep_empty)

    # @info "Remove Novices"
    # df=df |> @filter(_.ClassGroup != "Novice") |> DataFrame

    #constants for DCP
    r=Variable((nrow(df),run_groups),:Bin)


    #Set basic constraints
    constr = sum(r,dims=2) == 1 #Each class runs once
    #Bump Classes
    for bumpclass=unique(skipmissing(df.BumpClass))
        ind=[i for (i,x) in enumerate(Tables.rows(df)) if isequal(x.BumpClass,bumpclass) && x.Drivers < min_class_size]
        isempty(ind) && continue
        @info "Combining $(join(df.Class[ind],", "))"
        for (a,b) in zip(ind[1:end-1],ind[2:end])
            constr += r[a,:] == r[b,:]
        end
    end

    #Keep novice groups in alphabetical order
    let ind=findall(isequal("Novice"),df.ClassGroup)
        for (a,b) in zip(ind[1:end-1],ind[2:end])
            constr += sum(r[a,:] * (1:run_groups)) <= sum(r[b,:] * (1:run_groups))
        end
    end

    #Separate classes
    for (a,b) in separate_classes
        ind_a=findfirst(isequal(a),df.Class)
        ind_b=findfirst(isequal(b),df.Class)
        constr += r[ind_a,:] + r[ind_b,:] <= 1
    end

    # workers=sum(r .* df.Workers,dims=1)
    workers = r' * df.Workers
    drivers= r' * df.Drivers
    novice=let ind=findall(isequal("Novice"),df.ClassGroup)
        # df.Drivers[ind]' * r[ind,:]
        r[ind,:]' * df.Drivers[ind]
    end

    constr += maximum(drivers) - minimum(drivers) <= 6
    constr += maximum(novice) - minimum(novice) <= 8

    worker_diff = maximum(workers) - minimum(workers)
    driver_diff = maximum(drivers) - minimum(drivers)
    novice_diff = maximum(novice) - minimum(novice)

    # prob=maximize(minimum(workers),constr)
    prob=minimize(worker_diff*workerweight + driver_diff*driverweight + novice_diff*noviceweight,constr)

    (problem=prob,x=r,constraints=constr,df=df,workers=workers,drivers=drivers,novice=novice,satisfy=satisfy(prob.constraints))
end

function _class_counts(classes,entries;keep_empty=true)
    df=@from i in entries begin
        @group i by i.IndexedClass into g
        @select {Class=key(g),Drivers=length(g),Workers=sum(.!g.Exempt)}
        @collect DataFrame
    end
    df=join(df,classes,on=:Class,kind=:outer)
    df.Drivers=coalesce.(df.Drivers,0)
    df.Workers=coalesce.(df.Workers,0)
    if !keep_empty
        df=df |> @filter(_.Drivers != 0 || _.ClassGroup =="Novice") |> DataFrame
    end
    df |> @orderby(_.ClassGroup == "Novice") |> @thenby(_.ClassGroup) |> @thenby(_.Class) |> DataFrame
end

function full_solution(a)

    @assert a.problem.status ==:Optimal

    workers=Int.(evaluate(a.workers)) |> vec
    drivers=Int.(evaluate(a.drivers)) |> vec
    novices=Int.(evaluate(a.novice)) |> vec
    run_groups=size(a.x,2)

    df=map(zip(eachrow(evaluate(a.x)), Tables.rows(a.df))) do (x,row)
        i=findfirst(!iszero,x)
        (RunGroup=i,Class=row.Class,ClassGroup=row.ClassGroup, Drivers=row.Drivers,
        Workers=row.Workers ,Novice=row.ClassGroup=="Novice")
    end

    y=mapreduce(vcat, df |> @groupby(_.RunGroup)) do g
        mapreduce(vcat, g |> @groupby(_.Novice)) do g
            if first(g).Novice
                s=[x.Class[2] for x in g |> @filter(_.Novice)]
                class="Novice $(minimum(s))-$(maximum(s))"
                (RunGroup = first(g).RunGroup, Class=class, ClassGroup="Novice", Drivers=sum(g.Drivers),
                    Workers=sum(g.Workers), Novice=true)
            else
                g
            end
        end
    end

    y |> @orderby(_.RunGroup) |> @thenby(_.Novice) |> @select(:RunGroup,:Class,:ClassGroup, :Drivers, :Workers) |> DataFrame

end

function compact_solution(a)

    y = full_solution(a)

    @from i in y begin
        @group i by i.RunGroup into g
        @select {RunGroup=key(g), Workers=sum(g.Workers),
            Drivers=sum(x->x.ClassGroup!="Novice" ? x.Drivers : 0,g),
            Novice=sum(x->x.ClassGroup=="Novice" ? x.Drivers : 0, g)}
        @collect DataFrame
    end

end

end # module
