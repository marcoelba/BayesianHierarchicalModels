# Plot utils
using StatsPlots


function fdr_n_hist(class_distributions)

    plt_fdr = histogram(class_distributions.fdr_distribution, label=false, normalize=true)
    xlabel!("FDR", labelfontsize=15)
    vline!([mean(class_distributions.fdr_distribution)], color="red", label="Mean", linewidth=2)
    plt_n = histogram(class_distributions.n_selected_distribution, label=false, normalize=true)
    xlabel!("# Included Variables", labelfontsize=15)
    vline!([mean(class_distributions.n_selected_distribution)], color="red", label="Mean", linewidth=2)
    plt = plot(plt_fdr, plt_n)
    return plt
end


function scatter_sel_matrix(class_distributions; p0=false)
    mean_selection_matrix = mean(class_distributions.selection_matrix, dims=2)[:, 1]
    plt = scatter(mean_selection_matrix, label=false, markersize=3)
    if p0 > 0
        vline!([p0 + 1], label=false, color="green")
    end
    xlabel!("Covariates", labelfontsize=15)
    ylabel!("Inclusion probability", labelfontsize=15)
    return plt
end
