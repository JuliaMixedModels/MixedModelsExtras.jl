@testset "LMM" begin
    model = fit(MixedModel, @formula(reaction ~ 1 + (1 | subj)), dataset(:sleepstudy);
                progress)
    @test icc(model, :subj) == icc(model, [:subj]) == icc(model)
    @test icc(model, :subj) ≈ 0.37918288 rtol = 1e-6

    formula = @formula(rt_trunc ~ 1 + spkr * prec * load +
                                  (1 + spkr | subj) +
                                  (1 | item))
    model = fit(MixedModel, formula, dataset(:kb07); progress)
    @test icc(model, :subj) + icc(model, :item) ≈ icc(model)

    @testset "random slopes" begin
        # for a random-intercept model the Johnson (2014) average over observations
        # and the sum of the random-effect variances coincide
        intonly = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),
                      dataset(:sleepstudy); progress)
        @test icc(intonly, :subj; slopes=:mean) ≈ icc(intonly, :subj; slopes=:diagonal)

        slopemodel = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
                         dataset(:sleepstudy); progress)
        # with random slopes they differ, because the days covariate is not centered
        # and so scales up the average random-effects variance
        @test icc(slopemodel, :subj; slopes=:mean) ≈ 0.70909 rtol = 1e-4
        @test icc(slopemodel, :subj; slopes=:diagonal) ≈ 0.47736 rtol = 1e-4
        @test_throws ArgumentError icc(slopemodel, :subj; slopes=:bogus)
    end

    @testset "bootstrap" begin
        boot = parametricbootstrap(StableRNG(42), 100, model; progress)
        iccboot_subj = icc(boot, :subj; slopes=:diagonal)
        iccboot_item = icc(boot, :item; slopes=:diagonal)
        @test iccboot_subj + iccboot_item ≈ icc(boot; slopes=:diagonal)

        ci_subj = shortestcovint(iccboot_subj)
        ci_item = shortestcovint(iccboot_item)
        @test first(ci_subj) < icc(model, :subj; slopes=:diagonal) < last(ci_subj)
        @test first(ci_item) < icc(model, :item; slopes=:diagonal) < last(ci_item)

        # the Johnson (2014) average needs the model, because the bootstrap stores
        # only variances and correlations, not the random-effects model matrix
        @test_throws ArgumentError icc(boot, :subj)
        johnson_subj = icc(boot, :subj; model)
        ci_johnson = shortestcovint(johnson_subj)
        @test first(ci_johnson) < icc(model, :subj) < last(ci_johnson)
        @test johnson_subj != iccboot_subj
        # the item term is a random intercept, but its ICC still differs between the
        # two, because the denominator includes the random-slope subj term
        @test icc(boot, :item; model) != iccboot_item

        # for a model with only random intercepts the two are identical
        intonly = fit(MixedModel,
                      @formula(rt_trunc ~ 1 + spkr + (1 | subj) + (1 | item)),
                      dataset(:kb07); progress)
        intboot = parametricbootstrap(StableRNG(1), 20, intonly; progress)
        @test icc(intboot, :subj; model=intonly) ≈ icc(intboot, :subj; slopes=:diagonal)

        # Regression test: per-iteration values must be in sorted iteration
        # order so they align with boot.σ (residual σ, natural order).
        iters = sort!(unique(row.iter for row in Tables.rows(boot.σs)))
        σ²_subj_ref = [sum(abs2(row.σ)
                           for row in Tables.rows(boot.σs)
                           if row.iter == i && row.group == :subj; init=0.0)
                       for i in iters]
        σ²_all_ref = [sum(abs2(row.σ) for row in Tables.rows(boot.σs)
                          if row.iter == i; init=0.0)
                      for i in iters]
        @test iccboot_subj ≈ σ²_subj_ref ./ (abs2.(boot.σ) .+ σ²_all_ref)

        @testset "confint" begin
            @test iccboot_subj isa MixedModelsExtras.IccBootstrap
            @test iccboot_subj isa AbstractVector{Float64}

            @test confint(iccboot_subj) == shortestcovint(iccboot_subj)
            @test confint(iccboot_subj) == shortestcovint(iccboot_subj, 0.95)
            @test confint(iccboot_subj; level=0.8) == shortestcovint(iccboot_subj, 0.8)
            @test confint(icc(boot, [:subj, :item]; slopes=:diagonal)) ==
                  shortestcovint(icc(boot; slopes=:diagonal))

            lo, hi = confint(iccboot_subj; method=:equaltail)
            @test lo ≈ quantile(iccboot_subj, 0.025)
            @test hi ≈ quantile(iccboot_subj, 0.975)

            @test_throws ArgumentError confint(iccboot_subj; method=:bogus)

            # base MixedModels.jl `confint(::MixedModelBootstrap)` (no groups) is untouched
            @test Tables.istable(confint(boot))
        end

        @testset "show" begin
            str = sprint(show, MIME("text/plain"), iccboot_item)
            @test occursin("IccBootstrap", str)
            @test occursin("median", str)
            @test occursin("95% CI", str)
            @test occursin("100-element", str)
            # an empty bootstrap prints its header and nothing else
            @test occursin("0-element",
                           sprint(show, MIME("text/plain"),
                                  MixedModelsExtras.IccBootstrap(Float64[])))
        end

        @testset "conditional and groupmean" begin
            cond = icc(boot, :item; model, conditional=true)
            @test all(cond .< icc(boot, :item; model))
            # the per-iteration fixed-effects variances line up with boot.β
            @test length(cond) == length(iccboot_item)

            gm = icc(boot, :item; model, groupmean=true)
            @test all(gm .> icc(boot, :item; model))
            @test icc(boot, :item; k=2, slopes=:diagonal) isa
                  MixedModelsExtras.IccBootstrap

            # both need either the model or an explicit k
            @test_throws ArgumentError icc(boot, :item; conditional=true,
                                           slopes=:diagonal)
            @test_throws ArgumentError icc(boot, :item; groupmean=true,
                                           slopes=:diagonal)
        end
    end
end

@testset "conditional ICC" begin
    sleepstudy = dataset(:sleepstudy)
    model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy;
                progress)
    adjusted = icc(model, :subj)
    conditional = icc(model, :subj; conditional=true)
    # the conditional ICC adds var(Xβ) to the denominator, so it is always smaller
    # whenever the fixed effects explain anything at all
    @test conditional < adjusted
    σ²f = var(model.X * model.β)
    σ²subj = only(VarCorr(model).σρ[:subj].σ)^2
    @test conditional ≈ σ²subj / (σ²subj + varest(model) + σ²f)

    # with no fixed effects beyond an intercept the two coincide
    intercept = fit(MixedModel, @formula(reaction ~ 1 + (1 | subj)), sleepstudy; progress)
    @test icc(intercept, :subj) ≈ icc(intercept, :subj; conditional=true)
end

@testset "group-mean reliability" begin
    sleepstudy = dataset(:sleepstudy)
    model = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy;
                progress)
    icc1 = icc(model, :subj)
    # sleepstudy is balanced with 10 observations per subject
    @test icc(model, :subj; groupmean=true) ≈ 10 * icc1 / (1 + 9 * icc1)
    @test icc(model, :subj; groupmean=true, k=1) ≈ icc1
    # ICC(2) always exceeds ICC(1) for k > 1: averaging improves reliability
    @test icc(model, :subj; groupmean=true) > icc1

    # unequal group sizes use the harmonic mean
    unbalanced = DataFrame(sleepstudy)[1:170, :]
    umodel = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), unbalanced;
                 progress)
    counts = [count(==(s), unbalanced.subj) for s in unique(unbalanced.subj)]
    k̄ = length(counts) / sum(inv, counts)
    @test MixedModelsExtras._group_size(umodel, :subj) ≈ k̄
    @test 9 < k̄ < 10

    # a group-mean reliability is only defined for a single grouping variable
    kb07 = fit(MixedModel, @formula(rt_trunc ~ 1 + spkr + (1 | subj) + (1 | item)),
               dataset(:kb07); progress)
    @test_throws ArgumentError icc(kb07; groupmean=true)
    @test icc(kb07, :subj; groupmean=true) == icc(kb07, [:subj]; groupmean=true)
end

@testset "Binomial" begin
    cbpp = dataset(:cbpp)
    # suppress depwarn on wts vs weights
    model = @suppress fit(MixedModel, @formula((incid / hsz) ~ 1 + (1 | herd)),
                          cbpp, Binomial(); wts=float(cbpp.hsz), progress)
    @test icc(model, :herd) == icc(model, [:herd]) == icc(model)
    @test icc(model, :herd) ≈ 0.1668 atol = 0.0005
end

@testset "Bernoulli" begin
    contra = dataset(:contra)
    modelbern = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)),
                    contra, Bernoulli(); fast=true, progress)
    # force treating as a Binomial model
    # suppress depwarn on wts vs weights
    modelbin = @suppress fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)),
                             contra, Binomial(); fast=true, wts=ones(length(contra.dist)),
                             progress)
    # Bernoullis are a special case of binomial, so make sure they give the same answer
    @test icc(modelbern, Symbol("urban & dist")) ≈ icc(modelbin, Symbol("urban & dist"))

    @testset "observation-level variance" begin
        group = Symbol("urban & dist")
        # the theoretical (latent-scale) variance is the default for binomial families
        @test icc(modelbern, group; method=:theoretical) == icc(modelbern, group)
        p = MixedModelsExtras._mean_probability(modelbern)
        @test 0 < p < 1
        @test MixedModelsExtras._residual_variance(modelbern, :theoretical) ≈ π^2 / 3
        @test MixedModelsExtras._residual_variance(modelbern, :observation_level) ≈
              1 / (p * (1 - p))
        # 1/(p(1-p)) ≥ 4 > π²/3, so the observation-level variance is always the larger
        # of the two and the resulting ICC always the smaller
        @test icc(modelbern, group; method=:observation_level) < icc(modelbern, group)
        @test_throws ArgumentError icc(modelbern, group; method=:lognormal)
    end

    @testset "simulation VPC" begin
        group = Symbol("urban & dist")
        sim = icc(modelbern, group; method=:simulation, nsim=20_000, rng=StableRNG(42))
        @test 0 < sim < 1
        # the response-scale variance partition of Goldstein et al. (2002) is smaller
        # than the latent-scale one for binary data
        @test sim < icc(modelbern, group)
        # and it is stable across seeds
        @test sim ≈ icc(modelbern, group; method=:simulation, nsim=20_000,
                        rng=StableRNG(11)) atol = 0.01
        # simulation is not meaningful for a linear model, where the partition is exact
        lmm = fit(MixedModel, @formula(reaction ~ 1 + (1 | subj)), dataset(:sleepstudy);
                  progress)
        @test_throws ArgumentError icc(lmm; method=:simulation)
    end

    @testset "bootstrap" begin
        boot = parametricbootstrap(StableRNG(42), 100, modelbern; progress)
        @test_throws ArgumentError icc(boot)
        iccboot = icc(boot, Bernoulli())
        ci = shortestcovint(iccboot)
        @test first(ci) < icc(modelbern) < last(ci)
        @test iccboot ≈ icc(boot, Bernoulli(), Symbol("urban & dist"))
        @test confint(iccboot) == ci
        @test confint(icc(boot, Bernoulli(), Symbol("urban & dist"))) == ci

        # passing the model lets the observation-level methods be used, since they
        # need more than the family alone
        withmodel = icc(boot, Bernoulli(); model=modelbern)
        @test withmodel ≈ iccboot
        obslevel = icc(boot, Bernoulli(); model=modelbern, method=:observation_level)
        @test all(obslevel .< withmodel)
        @test_throws ArgumentError icc(boot, Bernoulli(); method=:observation_level)
        @test_throws ArgumentError icc(boot, Poisson(); method=:lognormal)
    end
end

@testset "Poisson" begin
    grouseticks = DataFrame(dataset(:grouseticks))
    grouseticks.ch = grouseticks.height .- mean(grouseticks.height)
    model = fit(MixedModel,
                @formula(ticks ~ 1 + year + ch + (1 | index) + (1 | brood) + (1 | location)),
                grouseticks, Poisson(); fast=true, progress)
    # the observation-level variance is the lognormal approximation of
    # Nakagawa et al. (2017); before v3 a residual variance of exactly 1 was used
    @test icc(model, :index) ≈ 0.20623907348997433 atol = 0.0005
    @test icc(model, [:index, :brood]) ≈ 0.593994569498534 atol = 0.0005
    @test icc(model, [:index, :brood, :location]) ≈ 0.816131958369578 atol = 0.0005
    @test icc(model, [:index, :brood, :location]) == icc(model)

    @testset "observation-level variance" begin
        # all three approximations are functions of the same expected rate
        λ = MixedModelsExtras._mean_rate(model)
        @test MixedModelsExtras._residual_variance(model, :delta) ≈ 1 / λ
        @test MixedModelsExtras._residual_variance(model, :lognormal) ≈ log1p(1 / λ)
        @test MixedModelsExtras._residual_variance(model, :trigamma) ≈ trigamma(λ)
        # :lognormal is the default for Poisson
        @test icc(model, :index; method=:lognormal) == icc(model, :index)
        # the approximations are close but not identical, and ordered
        @test icc(model, :index; method=:trigamma) <
              icc(model, :index; method=:delta) <
              icc(model, :index; method=:lognormal)
        @test_throws ArgumentError icc(model, :index; method=:theoretical)
    end

    @testset "simulation VPC" begin
        # the response-scale partition uses the Poisson variance function, var = μ
        sim = icc(model, :index; method=:simulation, nsim=10_000, rng=StableRNG(7))
        @test 0 < sim < 1
        @test sim ≈ icc(model, :index; method=:simulation, nsim=10_000,
                        rng=StableRNG(8)) atol = 0.05
        # conditional=true adds the response-scale fixed-effects variance
        @test icc(model, :index; method=:simulation, nsim=10_000, rng=StableRNG(7),
                  conditional=true) < sim
    end
end

@testset "links" begin
    # the theoretical observation-level variance is the variance of the latent
    # distribution the link implies
    contra = dataset(:contra)
    probit = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)), contra,
                 Bernoulli(), ProbitLink(); fast=true, progress)
    @test MixedModelsExtras._residual_variance(probit, :theoretical) == 1.0
    @test MixedModelsExtras._residual_variance(probit, :observation_level) > 0

    cloglog = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)), contra,
                  Bernoulli(), CloglogLink(); fast=true, progress)
    @test MixedModelsExtras._residual_variance(cloglog, :theoretical) ≈ π^2 / 6
    @test MixedModelsExtras._residual_variance(cloglog, :observation_level) > 0
end

@testset "Fallback" begin
    @test_throws(ArgumentError("Family TDist{Float64} currently unsupported, please file an issue."),
                 MixedModelsExtras._default_method(TDist(3)))
    @test_throws(ArgumentError("Family TDist{Float64} currently unsupported, please file an issue."),
                 MixedModelsExtras._residual_variance(TDist(3), IdentityLink(), :delta,
                                                      nothing))
    @test_throws(ArgumentError("Family TDist{Float64} currently unsupported, please file an issue."),
                 MixedModelsExtras._conditional_variance(TDist(3), 0.5))
end
