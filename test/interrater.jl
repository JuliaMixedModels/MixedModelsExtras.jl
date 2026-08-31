# Shrout & Fleiss (1979), Table 2: six targets rated by four judges.
# The published values are ICC(1,1) = 0.17, ICC(2,1) = 0.29, ICC(3,1) = 0.71.
const SF_RATINGS = [9 2 5 8
                    6 1 3 2
                    8 4 6 8
                    7 1 2 6
                    10 5 6 9
                    6 2 4 7]

const SF_LONG = DataFrame(; target=repeat(1:6; outer=4),
                          rater=repeat(1:4; inner=6),
                          score=float(vec(SF_RATINGS)))

@testset "Shrout & Fleiss (1979) Table 2" begin
    result = icc(SF_LONG; target=:target, rater=:rater, score=:score)
    @test result isa InterraterICCTable
    @test length(result) == 6

    # point estimates, as published
    @test result.ICC1.estimate ≈ 0.1657 atol = 5e-5
    @test result.ICCA1.estimate ≈ 0.2898 atol = 5e-5
    @test result.ICCC1.estimate ≈ 0.7148 atol = 5e-5
    @test result.ICC1k.estimate ≈ 0.4428 atol = 5e-5
    @test result.ICCAk.estimate ≈ 0.6201 atol = 5e-5
    @test result.ICCCk.estimate ≈ 0.9093 atol = 5e-5

    # confidence intervals, cross-checked against R's psych::ICC (printed to 4 decimals)
    @test all(isapprox.(confint(result.ICC1), (-0.1329, 0.7226); atol=5e-5))
    @test all(isapprox.(confint(result.ICCA1), (0.0188, 0.7611); atol=5e-5))
    @test all(isapprox.(confint(result.ICCC1), (0.3425, 0.9459); atol=5e-5))
    @test all(isapprox.(confint(result.ICCAk), (0.0711, 0.9272); atol=5e-5))

    @testset "naming" begin
        @test result.ICCA1.name == "ICC(A,1)"
        @test result.ICCA1.alias == "ICC(2,1)"
        # Shrout & Fleiss spellings index the same coefficients
        @test result[:ICC11] === result.ICC1
        @test result[:ICC21] === result.ICCA1
        @test result[:ICC31] === result.ICCC1
        @test result[:ICC2k] === result.ICCAk
        @test_throws KeyError result[:ICC99]
    end

    @testset "single coefficients" begin
        single = icc(SF_LONG; target=:target, rater=:rater, score=:score,
                     model=:twoway, type=:agreement, unit=:single)
        @test single isa InterraterICC
        @test single.estimate ≈ result.ICCA1.estimate
        @test icc(SF_LONG; target=:target, rater=:rater, score=:score,
                  type=:consistency).estimate ≈ result.ICCC1.estimate
        @test icc(SF_LONG; target=:target, rater=:rater, score=:score,
                  model=:oneway).estimate ≈ result.ICC1.estimate
        @test icc(SF_LONG; target=:target, rater=:rater, score=:score,
                  unit=:average).estimate ≈ result.ICCAk.estimate
    end

    @testset "variance components and bias test" begin
        c = result.ICCA1
        ms = c.meansquares
        # Liljequist et al. (2019) eq. 11
        @test c.varcomp.subject ≈ (ms.subject - ms.residual) / c.k
        @test c.varcomp.rater ≈ (ms.rater - ms.residual) / c.n
        @test c.varcomp.residual ≈ ms.residual
        # the within-subject mean square decomposes into rater and residual parts
        @test ms.within ≈
              ((c.k - 1) * ms.rater + (c.n - 1) * (c.k - 1) * ms.residual) /
              (c.n * (c.k - 1))

        # these judges differ systematically, which is exactly why ICC(C,1) is so
        # much larger than ICC(A,1) here
        @test c.biastest.F ≈ ms.rater / ms.residual
        @test c.biastest.p < 0.001
        @test result.ICCC1.estimate > result.ICCA1.estimate
    end

    @testset "level" begin
        wide = icc(SF_LONG; target=:target, rater=:rater, score=:score,
                   type=:consistency, level=0.99)
        narrow = icc(SF_LONG; target=:target, rater=:rater, score=:score,
                     type=:consistency, level=0.90)
        @test first(confint(wide)) < first(confint(narrow))
        @test last(confint(wide)) > last(confint(narrow))
        @test_throws ArgumentError confint(narrow; level=0.95)
        @test confint(narrow; level=0.90) == confint(narrow)
    end
end

@testset "estimators" begin
    anova = icc(SF_LONG; target=:target, rater=:rater, score=:score)
    reml = icc(SF_LONG; target=:target, rater=:rater, score=:score, estimator=:lmm)
    ml = icc(SF_LONG; target=:target, rater=:rater, score=:score, estimator=:lmm,
             reml=false)

    # on balanced data with an interior optimum, REML reproduces the ANOVA estimators
    # up to the tolerance of the optimizer
    for (a, r) in zip(anova, reml)
        @test a.estimate ≈ r.estimate rtol = 1e-5
    end
    # ML differs, because it does not correct for the estimation of the mean
    @test ml.ICCC1.estimate != reml.ICCC1.estimate

    # exact F intervals are only defined for the ANOVA estimator
    @test isnan(reml.ICCA1.lower)
    @test_throws ArgumentError confint(reml.ICCA1)
    @test reml.ICCA1.estimator === :lmm
end

@testset "unbalanced and incomplete data" begin
    incomplete = SF_LONG[1:(end - 1), :]
    @test_throws ArgumentError icc(incomplete; target=:target, rater=:rater, score=:score)
    # the mixed-model estimator handles it
    result = icc(incomplete; target=:target, rater=:rater, score=:score, estimator=:lmm)
    @test result isa InterraterICCTable
    @test 0 < result.ICCC1.estimate < 1

    duplicated = vcat(SF_LONG, SF_LONG[1:1, :])
    @test_throws ArgumentError icc(duplicated; target=:target, rater=:rater, score=:score)
end

@testset "negative estimates" begin
    # when the between-target mean square falls below the within-target mean square,
    # ICC(1) goes negative. Liljequist et al. (2019) note this is an unlucky sample,
    # not a meaningful value, so it is reported as-is rather than truncated at zero.
    noisy = DataFrame(; target=repeat(1:4; outer=3),
                      rater=repeat(1:3; inner=4),
                      score=float([1, 5, 1, 5, 5, 1, 5, 1, 1, 5, 1, 5]))
    result = icc(noisy; target=:target, rater=:rater, score=:score)
    @test result.ICC1.estimate < 0
    @test result.ICC1.estimate ≈
          (result.ICC1.meansquares.subject -
           result.ICC1.meansquares.within) /
          (result.ICC1.meansquares.subject +
           2 * result.ICC1.meansquares.within)
end

@testset "recovers known variance components" begin
    # simulate the Liljequist et al. (2019) measurement model
    #   x_ij = μ + r_i + c_j + v_ij
    # with known σ²r, σ²c and σ²v, then check that the estimates land close and that
    # the bias is detected
    rng = StableRNG(20250829)
    n, k = 200, 5
    σr, σc, σv = 3.0, 1.5, 1.0
    r = σr .* randn(rng, n)
    c = σc .* randn(rng, k)
    long = DataFrame(; target=repeat(1:n; outer=k),
                     rater=repeat(1:k; inner=n),
                     score=[10 + r[i] + c[j] + σv * randn(rng)
                            for j in 1:k for i in 1:n])
    result = icc(long; target=:target, rater=:rater, score=:score)

    vc = result.ICCA1.varcomp
    @test vc.subject ≈ σr^2 rtol = 0.2
    @test vc.residual ≈ σv^2 rtol = 0.2

    # the population values these estimate (Liljequist et al., eqs. for ρ₂A and ρ₂C)
    ρ_agreement = σr^2 / (σr^2 + σc^2 + σv^2)
    ρ_consistency = σr^2 / (σr^2 + σv^2)
    @test result.ICCA1.estimate ≈ ρ_agreement rtol = 0.25
    @test result.ICCC1.estimate ≈ ρ_consistency rtol = 0.1

    # with real rater bias present, consistency exceeds agreement and ICC(1) --
    # which cannot separate bias from noise -- estimates neither
    @test result.ICCC1.estimate > result.ICCA1.estimate
end

@testset "errors and show" begin
    @test_throws ArgumentError icc(SF_LONG; target=:nope, rater=:rater, score=:score)
    @test_throws ArgumentError icc(SF_LONG; target=:target, rater=:rater, score=:score,
                                   estimator=:bogus)
    @test_throws ArgumentError icc(SF_LONG; target=:target, rater=:rater, score=:score,
                                   model=:bogus)
    @test_throws ArgumentError icc(SF_LONG; target=:target, rater=:rater, score=:score,
                                   type=:bogus)
    @test_throws ArgumentError icc(SF_LONG; target=:target, rater=:rater, score=:score,
                                   unit=:bogus)
    @test_throws ArgumentError icc(SF_LONG; target=:target, rater=:rater, score=:score,
                                   level=1.5)
    @test_throws ArgumentError icc(42; target=:target, rater=:rater, score=:score)

    # at least two targets and two raters are needed
    tiny = DataFrame(; target=[1, 1], rater=[1, 2], score=[1.0, 2.0])
    @test_throws ArgumentError icc(tiny; target=:target, rater=:rater, score=:score)

    @testset "show" begin
        result = icc(SF_LONG; target=:target, rater=:rater, score=:score)
        table_str = sprint(show, MIME("text/plain"), result)
        @test occursin("ICC(A,1)", table_str)
        @test occursin("ICC(2,1)", table_str)
        @test occursin("rater bias", table_str)

        single_str = sprint(show, MIME("text/plain"), result.ICCA1)
        @test occursin("ICC(A,1) [ICC(2,1)]", single_str)
        @test occursin("variance components", single_str)
        @test occursin("95% CI", single_str)

        # the lmm estimator has no exact interval, so none is printed
        lmm_str = sprint(show, MIME("text/plain"),
                         icc(SF_LONG; target=:target, rater=:rater, score=:score,
                             estimator=:lmm, type=:consistency))
        @test !occursin("95% CI", lmm_str)
    end
end
