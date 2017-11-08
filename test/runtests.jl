using ChenZhou
using Base.Test
using DataFrames
using RDatasets
using GLM


# write your own tests here
@testset "Linear Model" begin
    form = dataset("datasets", "Formaldehyde")
    fml = @formula(OptDen ~ Carb)
    mf = ModelFrame(fml, form)
    mm = ModelMatrix(mf)
    y = mf.df[fml.lhs]
    md = Reg.LinearModel(mm.m, y)
    tmd = GLM.lm(fml, form)

    @testset "Fit" begin
        expect = coef(tmd)
        @test isapprox(expect, md.beta)
        @test isapprox(expect, coef(md))
        @test isapprox(expect, Reg.lm(fml, form) |> coef)


        @test isapprox(predict(tmd), fitted(md))
        @test isapprox(residuals(tmd), residuals(md))
    end

    @testset "ANOVA" begin
        expect = r2(tmd)
        actual = Reg.coef_of_determination(md)
        @test isapprox(expect, actual)
        @test isapprox(expect, r2(md))
    end

    @testset "Estimator" begin
        actual = stderr(md)
        expect = stderr(tmd)
        @test expect ≈ actual
    end

    @testset "Confidence interval" begin
        expect = confint(tmd)
        actual = confint(md)
        @test expect ≈ actual
    end
end
