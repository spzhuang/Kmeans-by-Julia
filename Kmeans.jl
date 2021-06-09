struct kmeans{T,S}
    dim1::T
    dim2::T
    dat::Vector{Vector{S}}
    function kmeans{T,S}(dim1::T,dim2::T) where {T<:Integer, S<:Real}
        dim1 = dim1
        dim2 = dim2
        dat = [rand(dim2) for i in 1:dim1]
        new(dim1,dim2,dat)
    end
end
kmeans(dim1::T,dim2::T) where {T<:Integer} = kmeans{T,Float64}(dim1,dim2)

function fit(km::kmeans;clu::Int)
   if km.dim1<clu
       error("contional: clu < km.dim1 ")
   end
   init_vec = rand(km.dat,clu)
   dist(x,y) = sum((x-y).^2)
   dist_array = [dist(i,j) for i in km.dat,j in init_vec]
   index = argmin(dist_array;dims=2)
   index = [i[2] for i  in index]
   inrange = collect(1:km.dim1)
   while true
       old_mid = copy(init_vec)
       for i in 1:clu
           curind = index.==i
           curind = filter(x->x>0,curind.*inrange)
           clus = km.dat[curind,:]
           curtemp = sum(clus,dims=1)/size(clus)[1]
           curtemp = curtemp[1:end]
           init_vec[i] = curtemp[1]
       end
       dist_array = [dist(i,j) for i in km.dat,j in init_vec]
       index = argmin(dist_array;dims=2)
       index = [i[2] for i  in index]
       condition = sum([dist(i[1],i[2]) for i in zip(old_mid,init_vec)])
       if condition<1e-3
           break
       end
   end
   res = Dict(zip(km.dat,index))
   return res
end

#  we can try runing kmeans
x = kmeans(10,4)
result = fit(x,clu=3)

# Dict{Vector{Float64}, Int64} with 10 entries:
#   [0.156319, 0.479576, 0.501251, 0.320281]  => 2
#   [0.0270444, 0.909711, 0.448424, 0.45363]  => 3
#   [0.611489, 0.82299, 0.423215, 0.336426]   => 1
#   [0.452124, 0.59486, 0.0913534, 0.145516]  => 2
#   [0.218026, 0.143118, 0.522476, 0.60786]   => 1
#   [0.0851472, 0.600737, 0.302818, 0.147116] => 2
#   [0.336521, 0.674593, 0.386824, 0.85873]   => 1
#   [0.64358, 0.448798, 0.150384, 0.886921]   => 1
#   [0.160526, 0.532066, 0.210685, 0.20071]   => 2
#   [0.0898102, 0.914038, 0.096469, 0.368438] => 3
