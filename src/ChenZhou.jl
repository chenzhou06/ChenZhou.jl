module ChenZhou

export Reg

module Reg
using DataFrames
import StatsBase

export LinearModel, coef


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

end # module reg


end # module
