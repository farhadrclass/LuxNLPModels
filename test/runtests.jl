using Test
using LuxNLPModels
using Lux, ComponentArrays, NLPModels
using MLUtils
using Random, Statistics, NNlib
using LinearAlgebra: norm, dot

# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------
rng = Random.MersenneTwister(42)

# Tiny model: fast to compile and evaluate
model = Chain(Dense(4 => 8, relu), Dense(8 => 2))
ps, st = Lux.setup(rng, model)
ps_cv  = ComponentArray(ps)

# Synthetic data: 32 samples, 2 batches of 16
x_all = randn(rng, Float32, 4, 32)
y_all = randn(rng, Float32, 2, 32)
loader = DataLoader((x_all, y_all); batchsize=16, shuffle=false)

loss_fn(ŷ, y) = mean(abs2, ŷ .- y)

# Convenience: fresh model with identical starting point each time
make_nlp() = LuxNLPModel(model, ps_cv, st, loader, loss_fn)

# ---------------------------------------------------------------------------
@testset "LuxNLPModels.jl" begin

    # ── Constructor ─────────────────────────────────────────────────────────
    @testset "Constructor" begin
        nlp = make_nlp()

        @test nlp.meta.nvar == length(ps_cv)

        # On the CPU device, x0 is a CPU vector. GPU tests use a CuArray x0.
        @test nlp.meta.x0 isa Vector{Float32}
        @test length(nlp.meta.x0) == length(ps_cv)

        # Initial parameter values are preserved
        @test nlp.meta.x0 ≈ Vector{Float32}(ps_cv)

        # Empty DataLoader must error at construction time
        empty_loader = DataLoader((zeros(Float32, 4, 0), zeros(Float32, 2, 0));
                                  batchsize=1)
        @test_throws Exception LuxNLPModel(model, ps_cv, st, empty_loader, loss_fn)
    end

    # ── obj ─────────────────────────────────────────────────────────────────
    @testset "obj" begin
        nlp = make_nlp()
        f   = obj(nlp, nlp.meta.x0)

        @test f isa Float32
        @test f > 0
        @test nlp.counters.neval_obj == 1

        # Second call increments counter again
        obj(nlp, nlp.meta.x0)
        @test nlp.counters.neval_obj == 2
    end

    # ── grad! ────────────────────────────────────────────────────────────────
    @testset "grad!" begin
        nlp = make_nlp()
        g   = similar(nlp.meta.x0)
        grad!(nlp, nlp.meta.x0, g)

        @test g isa Vector{Float32}
        @test length(g) == nlp.meta.nvar
        @test !all(iszero, g)           # non-trivial gradient
        @test nlp.counters.neval_grad == 1
    end

    # ── objgrad! ─────────────────────────────────────────────────────────────
    @testset "objgrad!" begin
        # obj and grad! called separately (reference)
        nlp_ref = make_nlp()
        g_ref   = similar(nlp_ref.meta.x0)
        f_ref   = obj(nlp_ref, nlp_ref.meta.x0)
        grad!(nlp_ref, nlp_ref.meta.x0, g_ref)

        # objgrad! in a single pass
        nlp_og = make_nlp()
        g_og   = similar(nlp_og.meta.x0)
        f_og, _ = objgrad!(nlp_og, nlp_og.meta.x0, g_og)

        @test f_og ≈ f_ref
        @test g_og ≈ g_ref

        # Both counters must be incremented
        @test nlp_og.counters.neval_obj  == 1
        @test nlp_og.counters.neval_grad == 1
    end

    # ── hprod! ───────────────────────────────────────────────────────────────
    @testset "hprod!" begin
        nlp = make_nlp()
        x0  = nlp.meta.x0
        v   = randn(rng, Float32, nlp.meta.nvar)
        hv  = similar(x0)

        hprod!(nlp, x0, v, hv)

        @test hv isa Vector{Float32}
        @test length(hv) == nlp.meta.nvar

        # Symmetry check: ⟨Hv, w⟩ ≈ ⟨v, Hw⟩  (finite-difference HvP)
        w  = randn(rng, Float32, nlp.meta.nvar)
        hw = similar(x0)
        nlp2 = make_nlp()
        hprod!(nlp2, x0, w, hw)
        @test dot(hv, w) ≈ dot(v, hw) rtol=1f-2

        @test nlp.counters.neval_hprod == 1
    end

    # ── minibatch_next_train! ────────────────────────────────────────────────
    @testset "minibatch_next_train!" begin
        nlp = make_nlp()

        X1, _ = nlp.current_batch        # batch 1  (samples 1–16)
        minibatch_next_train!(nlp)
        X2, _ = nlp.current_batch        # batch 2  (samples 17–32)
        @test X1 != X2                   # different data

        minibatch_next_train!(nlp)       # epoch ends → iterator restarts
        X3, _ = nlp.current_batch
        @test X3 == X1                   # back to batch 1
    end

end
