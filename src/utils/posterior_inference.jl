# Posterior Inference

module posterior_inference

    import Turing
    import Distributions
    import StatsPlots
    import Random
    import DataFrames

    function get_parameters_interval(turing_mcmc_chains; interval_type::String, alpha_prob::Float64=0.05)
        if (alpha_prob >= 1.) | (alpha_prob <= 0.)
            throw(error("alpha_prob MUST one in (0, 1)"))
        end
    
        if interval_type == "HPD"
            interval_matrix = Turing.hpd(turing_mcmc_chains, alpha=alpha_prob)
            df = DataFrames.DataFrame(interval_matrix)
            df[:, "Significative"] .= ""
            df[sign.(df[!, "lower"] .* df[!, "upper"]) .== 1., "Significative"] .= "*"
    
        elseif interval_type == "CI"
            # get lower and upper interval limits from the given alpha prob
            lower = alpha_prob / 2.
            upper = 1. - lower
            interval_matrix = Turing.quantile(turing_mcmc_chains, q=[lower, 0.5, upper])
            df = DataFrames.DataFrame(interval_matrix)
            # rename lower and upper bounds
            DataFrames.rename!(
                df,
                [string(lower*100) * "%" => "lower", string(upper*100) * "%" => "upper"]
            )
            df[:, "Significative"] .= ""
            df[sign.(df[!, "lower"] .* df[!, "upper"]) .== 1., "Significative"] .= "*"
        else
            throw(error("interval_type MUST be one of [CI, HDP]"))
        end
    
        return df
    
    end
    
end

