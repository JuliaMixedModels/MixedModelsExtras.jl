@testset "LMM" begin
    fm0 = fit(MixedModel, @formula(reaction ~ 0 + days + (1 | subj)), dataset(:sleepstudy);
              progress)
    fm1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), dataset(:sleepstudy);
              progress)
    fm2 = fit(MixedModel, @formula(reaction ~ 1 + days * days^2 + (1 | subj)),
              dataset(:sleepstudy); progress)

    ae_int = ArgumentError("VIF is only defined for models with an intercept term")
    ae_nterm = ArgumentError("VIF not meaningful for models with only one non-intercept term")
    @test_throws ae_int vif(fm0)
    @test_throws ae_nterm vif(fm1)

    mm = @view cor(modelmatrix(fm2))[2:end, 2:end]
    # this is a slightly different way to do the same
    # computation. I've verified that doing it this way
    # in R gives the same answers as car::vif
    # note that computing the same model from scratch in R
    # gives different VIFs ([70.41934, 439.10096, 184.00107]),
    # but that model is a slightly different fit than the Julia
    # one and that has knock-on effects
    @test isapprox(vif(fm2), diag(inv(mm)))

    # since these are all continuous preds, should be the same
    # but uses a very different computational method!
    @test isapprox(vif(fm2), gvif(fm2))
    # the scale factor is gvif^1/(2df)
    # so just sqrt.(vif) when everything is continuous
    @test isapprox(gvif(fm2; scale=true),
                   sqrt.(gvif(fm2)))
end

@testset "GVIF and RegrssionModel" begin
    duncan = rdataset("car", "Duncan")

    lm1 = lm(@formula(Prestige ~ 1 + Income + Education), duncan)
    @test termnames(lm1)[2] == coefnames(lm1)
    vif_lm1 = vif(lm1)

    # values here taken from car
    @test isapprox(vif_lm1, [2.1049, 2.1049]; atol=1e-5)
    @test isapprox(vif_lm1, gvif(lm1))

    lm2 = lm(@formula(Prestige ~ 1 + Income + Education + Type), duncan)
    @test termnames(lm2)[2] == ["(Intercept)", "Income", "Education", "Type"]
    @test isapprox(gvif(lm2), [2.209178, 5.297584, 5.098592]; atol=1e-5)
    @test isapprox(gvif(lm2; scale=true),
                   [1.486330, 2.301648, 1.502666]; atol=1e-5)
end
