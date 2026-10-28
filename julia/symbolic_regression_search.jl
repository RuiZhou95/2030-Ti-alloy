# 04_sr_search.jl —— SymbolicRegression.jl 0.24.5 参数扫描搜索脚本
# 用法: julia --project=julia_env --threads=8 04_sr_search.jl <target> <data_dir> <out_subdir>
# 环境变量: SR_POPULATIONS(默认33) SR_PARSIMONY(0.0032) SR_MAXSIZE(12)
# SR_NITER(默认500) SR_CYCLES(默认100)
using SymbolicRegression
using Random
using DelimitedFiles
using Printf
using Statistics

function main()
    target = ARGS[1]
    data_dir = ARGS[2]
    out_subdir = length(ARGS) >= 3 ? ARGS[3] : ""
    prefix = joinpath(data_dir, "sr_" * target)
    outdir = isempty(out_subdir) ? data_dir : joinpath(data_dir, "sr_scan", target, out_subdir)
    mkpath(outdir)
    output_path(stem) = isempty(out_subdir) ?
        joinpath(outdir, "sr_" * target * "_" * stem) : joinpath(outdir, stem)

    X_tr = readdlm(prefix * "_X_tr.csv", ',')
    y_tr = vec(readdlm(prefix * "_y_tr.csv", ','))
    X_te = readdlm(prefix * "_X_te.csv", ',')
    y_te = vec(readdlm(prefix * "_y_te.csv", ','))
    X_tr = permutedims(X_tr)
    X_te = permutedims(X_te)
    feature_names = readlines(prefix * "_features.txt")

    pops = parse(Int, get(ENV, "SR_POPULATIONS", "33"))
    pars = parse(Float64, get(ENV, "SR_PARSIMONY", "0.0032"))
    maxsz = parse(Int, get(ENV, "SR_MAXSIZE", "12"))
    niters = parse(Int, get(ENV, "SR_NITER", "500"))
    ncycles = parse(Int, get(ENV, "SR_CYCLES", "100"))

    options = Options(;
        binary_operators=(+, *, /, -),
        unary_operators=(square, cube, sqrt),
        populations=pops,
        maxsize=maxsz,
        parsimony=pars,
        fraction_replaced=0.00032,
        fraction_replaced_hof=0.036,
        crossover_probability=0.064,
        ncycles_per_iteration=ncycles,
        tournament_selection_p=0.86,
        verbosity=0,
        progress=false,
    )

    Random.seed!(42)
    hof = EquationSearch(X_tr, y_tr; niterations=niters, options=options)
    dominating = calculate_pareto_frontier(X_tr, y_tr, hof, options)

    n_tr = length(y_tr)
    pareto_rows = Vector{Tuple{Int64, Float64, Float64, String}}()
    best_idx = 1
    best_score = -Inf
    for (i, eq) in enumerate(dominating)
        cplx = compute_complexity(eq.tree, options)
        pred, _ = eval_tree_array(eq.tree, X_tr, options)
        pred = vec(pred)
        mse = sum((y_tr .- pred) .^ 2) / n_tr
        score = -mse - pars * cplx
        push!(pareto_rows, (cplx, mse, score, replace(string(eq.tree), "\n" => "")))
        if score > best_score
            best_score = score
            best_idx = i
        end
    end

    open(output_path("pareto.csv"), "w") do io
        println(io, "complexity,loss,score,formula")
        for (c, l, s, f) in sort(pareto_rows, by=r -> r[1])
            println(io, c, ",", l, ",", s, ",", f)
        end
    end

    best = dominating[best_idx]
    tree = best.tree
    cplx = compute_complexity(tree, options)
    pred_tr, _ = eval_tree_array(tree, X_tr, options)
    pred_te, _ = eval_tree_array(tree, X_te, options)
    pred_tr = vec(pred_tr)
    pred_te = vec(pred_te)
    mse_tr = sum((y_tr .- pred_tr) .^ 2) / n_tr
    r2_tr = 1 - sum((y_tr .- pred_tr) .^ 2) / sum((y_tr .- mean(y_tr)) .^ 2)
    r2_te = 1 - sum((y_te .- pred_te) .^ 2) / sum((y_te .- mean(y_te)) .^ 2)

    ftext = replace(string(tree), "\n" => "")
    for i in length(feature_names):-1:1
        ftext = replace(ftext, "x" * string(i) => feature_names[i])
    end

    open(output_path("pred_tr.csv"), "w") do io
        writedlm(io, pred_tr, ',')
    end
    open(output_path("pred_te.csv"), "w") do io
        writedlm(io, pred_te, ',')
    end
    open(output_path("best.json"), "w") do io
        write(io, "{\"complexity\": $cplx, \"loss\": $mse_tr, " *
            "\"score\": $(best_score), \"r2_train\": $r2_tr, \"r2_test\": $r2_te, " *
            "\"formula_text\": \"$(escape_string(ftext))\", " *
            "\"populations\": $pops, \"parsimony\": $pars, \"maxsize\": $maxsz, " *
            "\"niterations\": $niters, \"ncycles_per_iteration\": $ncycles}")
    end

    println("BEST formula: ", ftext)
    println("complexity=", cplx, " mse=", mse_tr)
    @printf("R2 train=%.4f test=%.4f\n", r2_tr, r2_te)
end

main()
