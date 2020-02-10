export simple_gradient, check_gradient

_single_derivative(f, v, dt::Real) = error("single derivative not implemented for type $(typeof(v))")

_increase_dt(s::Real, dt::Real) = s + dt
_increase_dt(s::Complex, dt) = Complex(real(s)+dt, imag(s)), Complex(real(s), imag(s)+dt)
_single_derivative(f, v::Real, dt::Real, fv::Real) = (f(_increase_dt(v, dt)) - fv) / dt
_single_derivative(f, v::Complex, dt::Real, fv::Real) = begin
    a, b = _increase_dt(v, dt)
    da = (f(a) - fv) / dt
    db = (f(b) - fv) / dt
    return da + db*im
end

_single_derivative(f, v::AbstractArray{<:Number}, dt::Real, fv::Real) = begin
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
        r[l] = _single_derivative(g, v[l], dt, fv)
        # tmp[l] = v[l]
    end
    return r
end

# _single_derivative(f, v::AbstractArray{<:Real}, dt::Real, fv::Real) = begin
#     x = copy(v)
#     tmp = similar(x)
#     for i in 1:length(v)
#         x[i] = _increase_dt(x[i], dt)
#         tmp[i] = (f(x) - fv) / dt
#         x[i] = v[i]
#     end 
#     return tmp
# end


# _single_derivative(f, v::AbstractArray{<:Complex}, dt::Real, fv::Real) = begin
#     x = copy(v)
#     tmp = similar(x)
#     for i in 1:length(v)
#         a, b = _increase_dt(x[i], dt)
#         x[i] = a
#         da = (f(x) - fv) / dt
#         x[i] = b
#         db = (f(x) - fv) / dt
#         tmp[i] = da + db*im
#         x[i] = v[i]
#     end 
#     return tmp
# end


function simple_gradient(f, args...; dt::Real=1.0e-6)
    r = []
    v0 = f(args...)
    # args = [args...]
    L = length(args)
    for l in 1:L
        g(x) = f(args[1:(l-1)]..., x, args[(l+1):L]...)
        push!(r, _single_derivative(g, args[l], dt, v0))
    end
    return (r...,)
end


function check_gradient(f, args...; dt::Real=1.0e-6, atol::Real=1.0e-4, verbose::Int=0)
    grad = collect_variables(gradient(f, args...))
    # tmpargs = [args...]
    tmpargs = args
    f2(x) = begin
        set_parameters!(x, tmpargs...)
        return f(tmpargs...)
    end
    grad1 = simple_gradient(f2, parameters(tmpargs...), dt=dt)[1]

    v = isapprox(grad, grad1, atol=atol)
    if (!v) && (verbose > 0)
        println("auto gradient is $grad.")
        println("finite difference gradient is $grad1")
    end
    return v
end



