

# function _rungekutta4order!(x, m, dt)
# 	k1 = dt .* (m * x)
# 	k2 = dt .* (m * (x + k1 ./ 2))
# 	k3 = dt .* (m * (x + k2 ./ 2))
# 	k4 = dt .* (m * (x + k3))
# 	@. x += (k1/6 + k2/3 + k3/3 + k4/6)
# end

# function evolve_mat_rungekutta(mat::AbstractMatrix, t::Number, x::AbstractVector, dt::Real=0.01)
# 	steps = round(Int, abs(t / dt))
# 	if steps == 0
# 	    dt = t
# 	    steps = 1
# 	else
# 		dt = t / steps
# 	end
# 	T = promote_type(eltype(mat), eltype(x), typeof(t))
# 	x_c = convert(Vector{T}, copy(x))
# 	for i in 1:steps
# 	    _rungekutta4order!(x_c, mat, dt)
# 	end
# 	return x_c
# end

function evolve_mat(mat::AbstractMatrix, t::Number, x::AbstractVector) 
	try
		# this seems to be more stable and faster
		abs_t = abs(t) * 10
		mat1 = mat * abs_t
		t1 = t / abs_t
		y, info = exponentiate(mat1, t1, x; maxiter=EXPM_MAXITER, ishermitian=true)

		# y, info = exponentiate(mat, t, x; maxiter=EXPM_MAXITER, ishermitian=true)
		(info.converged==1) || error("expm fail to converge.")
		return y
	catch
		# in the worst case, just do the brute force
		@warn "Krylov space exponentiate fail to converge."
		return exp(Matrix(mat) .* t) * x
		# return evolve_mat_rungekutta(mat, t, x)
	end
end 

# evolve_mat(mat::AbstractMatrix, t::Number, x::AbstractVector) = evolve_mat_rungekutta(mat, t, x)