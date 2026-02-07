@testset "LMM" begin
    m1 = fit(MixedModel,
             @formula(rt_trunc ~ 1 + spkr * prec * load +
                                 (1 + spkr + prec + load | subj) +
                                 (1 + spkr | item)),
             dataset(:kb07); progress)

    st = shrinkagetables(m1)
    for p in 1:3, grp in propertynames(st)
        sn = DataFrame(shrinkagenorm(m1; p)[grp])
        sts = DataFrame(st[grp])
        cols = names(sts, !in(["subj", "item"]))
        sts = transform(sts,
                        cols => ByRow((x...) -> norm(x, p)) => :shrinkage)
        @test all(isapprox.(sts.shrinkage, sn.shrinkage; atol=0.005))
    end

    @testset "_ranef error path" begin
        @suppress @test_throws PosDefException _ranef(m1, 1e12 .* m1.optsum.initial)
    end
end

@testset "GLMM" begin
    contra = dataset(:contra)
    modelbern = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)),
                    contra, Bernoulli(); fast=true, progress)
    st = shrinkagetables(modelbern)
    groups = propertynames(st)
    for p in 1:3, grp in groups
        sn = DataFrame(shrinkagenorm(modelbern; p)[grp])
        sts = DataFrame(st[grp])
        cols = names(sts, !in(string.(groups)))
        sts = transform(sts,
                        cols => ByRow((x...) -> norm(x, p)) => :shrinkage)
        @test all(isapprox.(sts.shrinkage, sn.shrinkage; atol=0.005))
    end

    @testset "_ranef error path" begin
        grouseticks = DataFrame(dataset(:grouseticks))
        model = fit(MixedModel,
                    @formula(ticks ~ 1 + year + height + (1 | index) + (1 | brood) +
                                     (1 | location)),
                    grouseticks, Poisson(); fast=true, progress)
        @suppress @test_throws ArgumentError _ranef(model, NaN .* model.optsum.initial)
    end
end
