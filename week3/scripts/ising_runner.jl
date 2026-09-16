using Printf
using Random: Xoshiro

function parse_cli_args(args)
    parsed = Dict{String, Any}(
        "every" => 0
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--update"
            i += 1; parsed["update"] = args[i]
        elseif arg == "--l"
            i += 1; parsed["l"] = parse(Int, args[i])
        elseif arg == "--t-from"
            i += 1; parsed["t-from"] = parse(Float64, args[i])
        elseif arg == "--t-to"
            i += 1; parsed["t-to"] = parse(Float64, args[i])
        elseif arg == "--t-step"
            i += 1; parsed["t-step"] = parse(Float64, args[i])
        elseif arg == "--discard"
            i += 1; parsed["discard"] = parse(Int, args[i])
        elseif arg == "--measure"
            i += 1; parsed["measure"] = parse(Int, args[i])
        elseif arg == "--seed"
            i += 1; parsed["seed"] = parse(UInt64, args[i])
        elseif arg == "--every"
            i += 1; parsed["every"] = parse(Int, args[i])
        elseif arg == "--out"
            i += 1; parsed["out"] = args[i]
        end
        i += 1
    end
    return parsed
end

@inline function sweep_metropolis!(spins::Vector{Int8}, nbrs::Matrix{Int32}, exp_table::Vector{Float64}, rng::Xoshiro, N::Int)
    accepted = 0
    de_total = 0
    d_spin_sum = 0
    @inbounds for _ in 1:N
        site = rand(rng, 1:N)
        s = spins[site]
        h = Int(spins[nbrs[1, site]]) + Int(spins[nbrs[2, site]]) + Int(spins[nbrs[3, site]]) + Int(spins[nbrs[4, site]])
        de = 2 * Int(s) * h
        pass = de <= 0 ? true : (de == 4 ? rand(rng) < exp_table[4] : rand(rng) < exp_table[5])
        if pass
            spins[site] = -s
            de_total += de
            d_spin_sum -= 2 * Int(s)
            accepted += 1
        end
    end
    return accepted, de_total, d_spin_sum
end

@inline function step_wolff!(spins::Vector{Int8}, nbrs::Matrix{Int32}, p_add::Float64, tag::Vector{UInt32}, queue::Vector{Int32}, cur_tag, rng::Xoshiro, N::Int)
    @inbounds begin
        seed_site = rand(rng, 1:N)
        s0 = spins[seed_site]
        tag[seed_site] = cur_tag
        queue[1] = seed_site
        head = 1
        tail = 1
        while head <= tail
            u = queue[head]
            head += 1
            for k in 1:4
                v = nbrs[k, u]
                if spins[v] == s0 && tag[v] != cur_tag
                    if rand(rng) < p_add
                        tag[v] = cur_tag
                        tail += 1
                        queue[tail] = v
                    end
                end
            end
        end
        cluster_size = tail

        de = 0
        for idx in 1:cluster_size
            u = queue[idx]
            for k in 1:4
                v = nbrs[k, u]
                if tag[v] != cur_tag
                    de += 2 * Int(s0) * Int(spins[v])
                end
            end
        end
        for idx in 1:cluster_size
            u = queue[idx]
            spins[u] = -s0
        end
        d_spin_sum = -2 * cluster_size * Int(s0)
        return cluster_size, de, d_spin_sum
    end
end

function run_simulation()
    cfg = parse_cli_args(ARGS)
    L = cfg["l"]
    N = L * L
    update_mode = cfg["update"]
    t_from = cfg["t-from"]
    t_to = cfg["t-to"]
    t_step = cfg["t-step"]
    discard = cfg["discard"]
    measure = cfg["measure"]
    seed_val = cfg["seed"]
    every_step = cfg["every"]
    out_dir = cfg["out"]

    mkpath(out_dir)

    t_grid = Float64[]
    cur_t = t_from
    while cur_t <= t_to + 1e-8 * max(abs(t_step), 1.0)
        push!(t_grid, round(cur_t, digits=6))
        cur_t += t_step
    end

    time_unit = update_mode == "metropolis" ? "sweep" : "cluster_flip"

    open(joinpath(out_dir, "run.json"), "w") do f
        println(f, "{")
        println(f, "  \"L\": ", L, ",")
        println(f, "  \"update\": \"", update_mode, "\",")
        println(f, "  \"t_grid\": [", join(t_grid, ", "), "],")
        println(f, "  \"discard\": ", discard, ",")
        println(f, "  \"measure\": ", measure, ",")
        println(f, "  \"seed\": ", seed_val, ",")
        println(f, "  \"sample_every\": 1,")
        println(f, "  \"time_unit\": \"", time_unit, "\"")
        println(f, "}")
    end

    nbrs = Matrix{Int32}(undef, 4, N)
    for y in 0:(L-1)
        for x in 0:(L-1)
            site = y * L + x + 1
            right = y * L + mod(x + 1, L) + 1
            up    = mod(y + 1, L) * L + x + 1
            left  = y * L + mod(x - 1 + L, L) + 1
            down  = mod(y - 1 + L, L) * L + x + 1
            nbrs[1, site] = right
            nbrs[2, site] = up
            nbrs[3, site] = left
            nbrs[4, site] = down
        end
    end

    spins = ones(Int8, N)
    energy = -2 * N
    spin_sum = N

    rng = Xoshiro(seed_val)

    series_f = open(joinpath(out_dir, "series.jsonl"), "w")
    spins_f = every_step > 0 ? open(joinpath(out_dir, "spins.jsonl"), "w") : nothing

    if update_mode == "metropolis"
        println("T\tmean |M|\tacceptance")
    else
        println("T\tmean |M|\tmean cluster size")
    end

    cumulative_sweep = 0
    tag = zeros(UInt32, N)
    cur_tag = UInt32(1)
    queue = Vector{Int32}(undef, N)

    for T in t_grid
        if update_mode == "metropolis"
            exp_table = [1.0, 1.0, 1.0, exp(-4.0 / T), exp(-8.0 / T)]
            accepted_total = 0

            for _ in 1:discard
                acc, de, d_s = sweep_metropolis!(spins, nbrs, exp_table, rng, N)
                accepted_total += acc
                energy += de
                spin_sum += d_s
                cumulative_sweep += 1
            end

            sum_abs_m = 0.0
            for step in 1:measure
                acc, de, d_s = sweep_metropolis!(spins, nbrs, exp_table, rng, N)
                accepted_total += acc
                energy += de
                spin_sum += d_s
                cumulative_sweep += 1

                m = spin_sum / N
                e = energy / N
                sum_abs_m += abs(m)

                @printf(series_f, "{\"L\":%d,\"T\":%g,\"sweep\":%d,\"M\":%.6f,\"E\":%.6f}\n", L, T, step, m, e)

                if every_step > 0 && step % every_step == 0
                    print(spins_f, "{\"L\":", L, ",\"T\":", T, ",\"sweep\":", cumulative_sweep, ",\"m\":", round(m, digits=6), ",\"spins\":[")
                    for s_idx in 1:N
                        if s_idx > 1; print(spins_f, ","); end
                        print(spins_f, spins[s_idx])
                    end
                    println(spins_f, "]}")
                end
            end

            mean_abs_m = sum_abs_m / measure
            acc_rate = accepted_total / ((discard + measure) * N)
            @printf("%g\t%.4f\t%.4f\n", T, mean_abs_m, acc_rate)

        elseif update_mode == "wolff"
            p_add = 1.0 - exp(-2.0 / T)
            total_cluster_size = 0

            for _ in 1:discard
                cur_tag += 1
                if cur_tag == typemax(UInt32)
                    fill!(tag, 0)
                    cur_tag = UInt32(1)
                end
                c, de, d_s = step_wolff!(spins, nbrs, p_add, tag, queue, cur_tag, rng, N)
                energy += de
                spin_sum += d_s
                cumulative_sweep += 1
            end

            sum_abs_m = 0.0
            for step in 1:measure
                cur_tag += 1
                if cur_tag == typemax(UInt32)
                    fill!(tag, 0)
                    cur_tag = UInt32(1)
                end
                c, de, d_s = step_wolff!(spins, nbrs, p_add, tag, queue, cur_tag, rng, N)
                total_cluster_size += c
                energy += de
                spin_sum += d_s
                cumulative_sweep += 1

                m = spin_sum / N
                e = energy / N
                sum_abs_m += abs(m)

                @printf(series_f, "{\"L\":%d,\"T\":%g,\"sweep\":%d,\"M\":%.6f,\"E\":%.6f,\"cluster_size\":%d}\n", L, T, step, m, e, c)

                if every_step > 0 && step % every_step == 0
                    print(spins_f, "{\"L\":", L, ",\"T\":", T, ",\"sweep\":", cumulative_sweep, ",\"m\":", round(m, digits=6), ",\"spins\":[")
                    for s_idx in 1:N
                        if s_idx > 1; print(spins_f, ","); end
                        print(spins_f, spins[s_idx])
                    end
                    println(spins_f, "]}")
                end
            end

            mean_abs_m = sum_abs_m / measure
            mean_c = total_cluster_size / measure
            @printf("%g\t%.4f\t%.4f\n", T, mean_abs_m, mean_c)
        end
    end

    close(series_f)
    if spins_f !== nothing
        close(spins_f)
    end
end

run_simulation()
