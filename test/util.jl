

using FiniteDifferences


fdm_gradient(f, v; kwargs...) = error("fdm_gradient not implemented for type $(typeof(v))")

fdm_gradient(f, v::Real; order::Int=5, kwargs...) = central_fdm(order, 1; kwargs...)(f, v)
fdm_gradient(f, v::Complex; order::Int=5, kwargs...) = begin
    f_real(vr::Real) = f(complex(vr, imag(v)))
    f_imag(vi::Real) = f(complex(real(v), vi))
    m = central_fdm(order, 1; kwargs...)
    da = m(f_real, real(v))
    db = m(f_imag, imag(v))
    return complex(da, db)
end

fdm_gradient(f, v::AbstractArray; kwargs...) = begin
    # x = copy(v)
    tmp = copy(v)
    r = similar(v)
    for l in 1:length(v)
        g(s) = begin
            tmp[l] = s
            d = f(tmp)
            tmp[l] = v[l]
            return d
        end
        r[l] = fdm_gradient(g, v[l]; kwargs...)
        # tmp[l] = v[l]
    end
    return r
end


function fdm_gradient(f, args...; kwargs...)
    r = []
    v0 = f(args...)
    # args = [args...]
    L = length(args)
    for l in 1:L
        g(x) = f(args[1:(l-1)]..., x, args[(l+1):L]...)
        push!(r, fdm_gradient(g, args[l]; kwargs...))
    end
    return (r...,)
end

# collect_variables_impl!(a::Vector, b::Nothing) = nothing
# collect_variables_impl!(a::Vector, b::Number) = push!(a, b)
# function collect_variables_impl!(a::Vector, b::AbstractArray) 
#     for item in b
#         collect_variables_impl!(a, item)
#     end
# end

# function collect_variables_impl!(a::Vector, b::AbstractDict)
#     for (k, v) in b
#         collect_variables_impl!(a, v)
#     end
# end

# function collect_variables_impl!(a::Vector, b::NamedTuple)
#     for v in b
#         collect_variables_impl!(a, v)
#     end
# end

# function collect_variables_impl!(a::Vector, b::Tuple)
#     for v in b
#         collect_variables_impl!(a, v)
#     end
# end

# """
#     collect_variables(args...)
# Collect variables from args...
# """
# function collect_variables(args...)
#     a = Number[]
#     for item in args
#         collect_variables_impl!(a, item)
#     end
#     return a
# end



# """
#     check_gradient(f, args...; verbosity::Int=0)
# Check the gradient of a function f with arguments args...
# """
# function check_gradient(f, args...; verbosity::Int=0)
#     grad = collect_variables(gradient(f, args...))
#     tmpargs = collect(args)
#     f2(x) = begin
#         set_parameters!(x, tmpargs)
#         return f(tmpargs...)
#     end
#     grad1 = simple_gradient(f2, collect_variables(tmpargs), dt=dt)[1]

#     diffmax = maximum(abs.(grad - grad1))
#     v = diffmax <= atol
#     if (!v) && (verbosity > 0)
#         println("largest difference is $diffmax.")
#         println("auto gradient is $grad.")
#         println("---------------------------------------------------------------------")
#         println("finite difference gradient is $grad1")
#     end
#     return v
# end