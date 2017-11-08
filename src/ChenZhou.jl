module ChenZhou

export Reg

module Reg
using DataFrames
using Distributions
import StatsBase

export LinearModel, coef, coef_of_determination


struct LinearModel
    X::AbstractMatrix
    y::AbstractArray
    beta::AbstractArray
    function LinearModel(X, y)
        beta = inv(X'X) * X' * y
        new(X, y, beta)
    end
end

lm(fml::Formula, df::DataFrame) = begin
    mf = ModelFrame(fml, df)
    mm = ModelMatrix(mf)
    LinearModel(mm.m, mf.df[fml.lhs])
end


StatsBase.coef(obj::LinearModel) = obj.beta

"""
    residual_maker(m)

Compute the residual maker matrix of a linear model, which produces
the vector of least squares residuals when it pre-multiples any vector
y.

"""
residual_maker(m::LinearModel) = begin
    # I - X(X'X)^-1 X'
    # ^   ^^^^^^^^^^^
    #     projection matrix
    pm = projection_matrix(m)
    n = size(pm)[1]
    eye(n) - pm
end # TODO: unittest

"""
    projection_matrix(m)

Compute the projection matrix which makes predicted y when it is pre-multipled y.

"""
projection_matrix(m::LinearModel) = begin
    X = m.X
    X * inv(X'X) * X'
end # TODO: unittest

StatsBase.fitted(m::LinearModel) = projection_matrix(m) * m.y
StatsBase.residuals(m::LinearModel) = residual_maker(m) * m.y
StatsBase.dof_residual(m::LinearModel) = begin
    n, K = size(m.X)
    n - K
end

total_sum_of_squares(m::LinearModel) = abs2.(m.y - mean(m.y)) |> sum
error_sum_of_squares(m::LinearModel) = begin
    e = residuals(m)
    abs2.(e - mean(e)) |> sum
end
coef_of_determination(m::LinearModel) = 1 - error_sum_of_squares(m) / total_sum_of_squares(m)
StatsBase.r2(m::LinearModel) = coef_of_determination(m)

StatsBase.stderr(m::LinearModel) = begin
    e = residuals(m)
    s2 = (e' * e) / dof_residual(m)
    sqrt.(s2 * inv(m.X' * m.X) |> diag)
end

StatsBase.confint(m::LinearModel, level::Real=0.95) = begin
    stderr(m) * quantile(
        TDist(dof_residual(m)),
        (1. - level) / 2.
    ) * [1.  -1.] +
        hcat(coef(m), coef(m))
end


end # module reg


end # module
