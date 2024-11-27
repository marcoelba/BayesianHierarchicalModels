# PCA as Autoencoder
using Flux
using MultivariateStats
using Distributions
using Random
using StatsPlots
using LinearAlgebra


n = 100
p = 4

X_dist = Normal(0., 1.)
Z = rand(X_dist, n, p)
X = zeros(n, p)
X[:, 1] = Z[:, 1]
X[:, 2] = X[:, 1] * 0.8 .+ Z[:, 2] .* 0.5
X[:, 3] = Z[:, 3]
X[:, 4] = Z[:, 4]
cor(X)
Xtr = X'

scatter(X[:, 1], X[:, 2])

pca = fit(PCA, Xtr; maxoutdim=3)
X_pred = X * pca.proj
cor(X_pred)

scatter(X_pred[:, 1], X_pred[:, 2])

X_pred * pca.proj'

pca.proj'
inv(pca.proj' * pca.proj) * pca.proj'


# Flux code
Xtrain = Xtr[:, 1:50]
Xtest = Xtr[:, 51:100]

Xtrain = Float32.(Xtrain)

# Define the model
model = Chain(
    Dense(p => 10, Flux.identity, bias=false),  # Input layer with 2 inputs, hidden layer with 10 neurons
    Dense(10 => p, Flux.identity, bias=false)      # Output layer with 1 output (sigmoid activation)
)
model(Xtrain)

# Define loss function and optimizer
loss(model, x) = Flux.Losses.mse(model(x), x)
loss(model, Xtrain)
optimizer = Flux.setup(Adam(), model)

# Training loop
epochs = 1000
for epoch in 1:epochs
    
    Flux.train!(loss, model, [Xtrain], optimizer)  # Transpose x for correct shape
    
    # Print loss every 10 epochs
    if epoch % 10 == 0
        println("Epoch $epoch: Loss = $(loss(model, Xtrain))")
    end
end

Flux.params(model)[1]
pca.proj'


X_pred_AE = X * Flux.params(model)[1]'
scatter(X_pred_AE[:, 1], X_pred_AE[:, 2])

X_pred_PCA = X * pca.proj
scatter!(X_pred_PCA[:, 1], X_pred_PCA[:, 2])

X_pred_PCA = Xtest' * pca.proj
scatter(X_pred_PCA[:, 1], Xtest'[:, 1])

X_pred_AE = model(Xtrain)'
scatter(X_pred_AE[:, 1], Xtrain[1, :])
Xtest_pred_AE = model(Xtest)'
scatter(Xtest_pred_AE[:, 1], Xtest[1, :])


# Regression case
abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))

n = 200
p = 10
prop_non_zero = 0.5
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.8

random_seed = Random.seed!(324)
data_dict = generate_linear_model_data(
    n_individuals=n,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed
)

n_train = Int(n / 2)
n_test = n - n_train
train_ids = sample(1:n, n_train, replace=false)
test_ids = setdiff(1:n, train_ids)

X_train = data_dict["X"][train_ids, :]
X_test = data_dict["X"][test_ids, :]
y_train = data_dict["y"][train_ids]
y_test = data_dict["y"][test_ids]


subset_dim = 3

"""
    PCA
"""
pca = fit(PCA, X_train'; maxoutdim=subset_dim)
X_fit_pca = X_train * pca.proj

# Regression
beta_ols = inv(X_fit_pca' * X_fit_pca) * X_fit_pca' * y_train

y_train_pred_pca = X_fit_pca * beta_ols
mean((y_train .- y_train_pred_pca).^2)

X_test_fit_pca = X_test * pca.proj
y_test_pred_pca = X_test_fit_pca * beta_ols
mean((y_test .- y_test_pred_pca).^2)


"""
    AE
"""
ae_dim = p

model = Chain(
    Dense(p => ae_dim, Flux.identity, bias=false),  # Input layer with 2 inputs, hidden layer with 10 neurons
    Dense(ae_dim => p, Flux.identity, bias=false)      # Output layer with 1 output (sigmoid activation)
)


# Define loss function and optimizer
loss(model, x) = Flux.Losses.mse(model(x), x)
optimizer = Flux.setup(Adam(), model)

# Training loop
epochs = 2000
train_loss_trace = []
test_loss_trace = []

for epoch in 1:epochs
    
    Flux.train!(loss, model, [X_train'], optimizer)  # Transpose x for correct shape
    
    # Print loss every 10 epochs
    push!(train_loss_trace, loss(model, X_train'))
    push!(test_loss_trace, loss(model, X_test'))

end
plot(train_loss_trace)
plot!(test_loss_trace)

X_fit_ae = X_train * Flux.params(model)[1]'

# Regression
beta_ols_ae = inv(X_fit_ae' * X_fit_ae) * X_fit_ae' * y_train
y_train_pred_ae = X_fit_ae * beta_ols_ae
mean((y_train .- y_train_pred_ae).^2)

X_test_fit_ae = X_test * Flux.params(model)[1]'
y_test_pred_ae = X_test_fit_ae * beta_ols_ae
mean((y_test .- y_test_pred_ae).^2)


"""
    AE + Regression
"""
using Fluxperimental

subset_dim = 3
ae_dim = subset_dim

# AE model
encoder = Dense(p => ae_dim, Flux.identity, bias=false)
decoder = Dense(ae_dim => p, Flux.identity, bias=false)

ae_model(x) = decoder(encoder(x))
ae_model(X_train')

ae_loss(x_pred, x) = Flux.Losses.mse(x_pred, x)
ae_loss(ae_model(X_train'), X_train')

# Regression Model
linear = Dense(ae_dim => 1, Flux.identity, bias=true)

linear_model(x) = linear(encoder(x))
linear_model(X_train')

lin_loss(y_pred, y) = Flux.Losses.mse(y_pred, y)
lin_loss(linear_model(X_train'), y_train')

# Total model and loss
joint_model = Chain(
    encoder,
    Fluxperimental.Split(
        decoder,
        linear
    )
)
pred = joint_model(X_train')
pred[1]
pred[2]

joint_loss(model_pred, x, y) = ae_loss(model_pred[1], x) + lin_loss(model_pred[2], y)
joint_loss(joint_model(X_train'), X_train', y_train')

# optimizer
optimizer = Flux.setup(Adam(), joint_model)

# Training loop
epochs = 2000
train_loss_trace = []
test_loss_trace = []
test_loss_ae_trace = []
test_loss_lin_trace = []

for epoch = 1:epochs

    train_loss, grads = Flux.withgradient(joint_model) do m
        # Any code inside here is differentiated.
        # Evaluation of the model and loss must be inside!
        result = m(X_train')
        joint_loss(result, X_train', y_train')
    end

    # Save the loss from the forward pass. (Done outside of gradient.)
    push!(train_loss_trace, train_loss)

    test_result = joint_model(X_test')
    
    push!(test_loss_ae_trace, ae_loss(test_result[1], X_test'))
    push!(test_loss_lin_trace, lin_loss(test_result[2], y_test'))

    # push!(
    #     test_loss_trace,
    #     joint_loss(test_result, X_test', y_test')
    # )

    Flux.update!(optimizer, joint_model, grads[1])

end

plot(train_loss_trace, label="train")
plot!(test_loss_trace)
plot!(test_loss_ae_trace, label="AE test")
plot!(test_loss_lin_trace, label="LIN test")

Flux.params(joint_model)[1]'
Flux.params(joint_model)[2]'
Flux.params(joint_model)[3]'

X_fit_ae = X_train * Flux.params(joint_model)[1]'

# Regression
beta_ols_ae = inv(X_fit_ae' * X_fit_ae) * X_fit_ae' * y_train
y_train_pred_ae = X_fit_ae * beta_ols_ae
mean((y_train .- y_train_pred_ae).^2)

X_test_fit_ae = X_test * Flux.params(joint_model)[3]'
y_test_pred_ae = X_test_fit_ae * beta_ols_ae
mean((y_test .- y_test_pred_ae).^2)
