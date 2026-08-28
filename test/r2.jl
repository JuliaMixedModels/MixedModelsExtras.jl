function _adjust(r2, model)
    n = nobs(model)
    p = dof(model)
    return 1 - (1 - r2) * (n - 1) / (n - p)
end

@testset "LMM" begin
    # can't use intercept-only FE because this leads to
    # a constant response in the conditional=false case and a NaN correlation
    model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),
                dataset(:sleepstudy); progress)

    @test r2(model; conditional=true) ≈ cor(response(model), fitted(model))^2
    @test r2(model; conditional=false) ≈ cor(response(model), model.X * model.β)^2

    @test adjr2(model; conditional=true) ≈ _adjust(r2(model; conditional=true), model)
    @test adjr2(model; conditional=false) ≈ _adjust(r2(model; conditional=false), model)
end
