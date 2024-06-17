# Mirror statistic
using Turing
using Random


function get_t(mirror_coeffs; fdr_q=0.1)
    
    optimal_t = 0
    t = 0
    
    for t in sort(abs.(mirror_coeffs))
        n_left_tail = sum(mirror_coeffs .< -t)
        n_right_tail = sum(mirror_coeffs .> t)
        n_right_tail = ifelse(n_right_tail .> 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail
    
        if fdp <= fdr_q
            optimal_t = t
            break
        end
    end
    return optimal_t    
end


function posterior_mirror_stat(posterior_samples; fdr_target=0.1)
    p = size(posterior_samples)[1]
    mc_samples = Int(size(posterior_samples)[2] / 2)
    posterior_ms_coefs = zeros(p, mc_samples)

    for param in range(1, p)
        mirror_coeffs = abs.(posterior_samples[param, 1:mc_samples] .+ posterior_samples[param, (mc_samples + 1):(mc_samples * 2)]) .-
            abs.(posterior_samples[param, 1:mc_samples] .- posterior_samples[param, (mc_samples + 1):(mc_samples * 2)])
        posterior_ms_coefs[param, :] = mirror_coeffs
    end

    # Get inclusion for each MC sample
    posterior_ms_inclusion = zeros(size(posterior_ms_coefs))
    for mc in range(1, mc_samples)
        opt_t_mc = get_t(posterior_ms_coefs[:, mc], fdr_q=fdr_target)
        posterior_ms_inclusion[:, mc] = posterior_ms_coefs[:, mc] .> opt_t_mc
    end

    return Dict("posterior_ms_coefs" => posterior_ms_coefs, "posterior_ms_inclusion" => posterior_ms_inclusion)
end
