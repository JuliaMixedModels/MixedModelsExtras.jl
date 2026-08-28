ssfit(f) = fit(MixedModel, f, dataset(:sleepstudy); progress)

a = ssfit(@formula(reaction ~ 1 + days + (1 | subj)))
b = ssfit(@formula(reaction ~ 1 + days + (1 + days | subj)))
c = ssfit(@formula(reaction ~ 1 + (1 | subj)))
d = ssfit(@formula(reaction ~ 1 + (1 + days | subj)))

tbl = ictable(a, b, c, d)

@test issubset([:model, :DoF, :ΔAIC, :ΔAICc, :ΔBIC],
               Tables.columnnames(tbl))

@test Tables.rowcount(tbl) == 4

@test all(>=(0), Tables.getcolumn(tbl, :ΔAIC))
@test all(>=(0), Tables.getcolumn(tbl, :ΔAICc))
@test all(>=(0), Tables.getcolumn(tbl, :ΔBIC))
@test all(>=(0), Tables.getcolumn(tbl, :DoF))
