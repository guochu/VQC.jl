# distance.jl — 态间距离

_distance2(x, y) = abs(real(dot(x, x)) + real(dot(y, y)) - 2 * real(dot(x, y)))
_distance(x, y) = sqrt(_distance2(x, y))
