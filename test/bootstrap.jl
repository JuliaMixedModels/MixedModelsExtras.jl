sleepstudy = dataset(:sleepstudy)
fm0 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),
          sleepstudy; progress)
fm1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
          sleepstudy; progress)
fmzc = fit(MixedModel, @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
           sleepstudy; progress)

# fm0 ⊂ fmzc ⊂ fm1, in ascending dof order, so the analytic LRT is well defined
# and gives us a reference to check the bootstrap version against.
lrt = likelihoodratiotest(fm0, fmzc, fm1)
d0, dzc, d1 = deviance(fm0), deviance(fmzc), deviance(fm1)

boot = bootstrap_lrt(StableRNG(42), 200, fm0, fmzc, fm1;
                     progress, optsum_overrides=(; maxfeval=500))

@testset "bookkeeping matches the analytic LRT" begin
    @test boot.dof == lrt.dof
    @test all(isapprox.(boot.deviance, lrt.deviance))
    @test boot.formulas == lrt.formulas
    @test boot.linear
end

@testset "p-values track the sign of the analytic LRT" begin
    @test isnan(boot.pvalues[1])
    # fm0 vs fmzc: huge deviance drop, analytic p ≈ 9e-11
    @test boot.pvalues[2] < 0.05
    # fmzc vs fm1: negligible deviance drop, analytic p ≈ 0.80
    @test boot.pvalues[3] > 0.3
end

@testset "original fits are restored after bootstrapping" begin
    @test deviance(fm0) ≈ d0
    @test deviance(fmzc) ≈ dzc
    @test deviance(fm1) ≈ d1
end

@testset "bookkeeping is independent of the order of ms..." begin
    # only the identity of m0 (the data-generating model) should matter;
    # re-sorting by dof for reporting must not depend on argument order
    boot2 = bootstrap_lrt(StableRNG(42), 200, fm0, fm1, fmzc; progress)
    @test boot2.dof == boot.dof
    @test boot2.formulas == boot.formulas
    @test boot2.deviance == boot.deviance
    @test isequal(boot2.pvalues, boot.pvalues)
end
