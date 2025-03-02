def filter(Sim, sampled_data_x, PPD_1, Number_Nodes, layers, kern_1_1, kern_1, std_dev, dis_size, n_mc_samples=100):
    t_obs_1 = abs(sampled_data['P2_bar'] - sampled_data['P1_bar']) ** 1
    U_obs = sampled_data['Filter_SL_Level'] / 100
    t_test_1 = (abs(Sim['P2_bar'] - Sim['P1_bar'])) * 1
    grad = (Sim['P1_bar']) / dis_size
    t_test_1_X = grad + t_test_1
    t_test_1 = np.linspace(t_test_1, t_test_1_X, dis_size)
    t_test_1 = pd.DataFrame(t_test_1).T

    torch.manual_seed(123)
    pinn = ModelFCN(1, 1, Number_Nodes, layers)
    t_physics = torch.linspace(abs(PPD_1.P2_bar - PPD_1.P1_bar).min(),
                               abs(PPD_1.P2_bar - PPD_1.P1_bar).max(), 300).reshape(-1, 1).requires_grad_(True)
    mu_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    optimiser = torch.optim.NAdam(list(pinn.parameters()) + [mu_1], lr=kern_1_1)

    loss_values = []
    loss3_values = []
    loss2_values = []

    predictions = []
    uncertainties = []

    for i in range(pinn_range_1):
        optimiser.zero_grad()
        lambda1 = kern_1
        t_obs_tensor = torch.tensor(t_obs_1.values, dtype=torch.float32).reshape(-1, 1)
        u_obs_tensor = torch.tensor(U_obs.values, dtype=torch.float32).reshape(-1, 1)
        t_test_tensor = torch.tensor(t_test_1.values, dtype=torch.float32).reshape(-1, 1)

        e = pinn(t_obs_tensor)
        loss2 = torch.mean((e - u_obs_tensor) ** 2)
        e_physics = pinn(t_physics)

        loss3 = torch.mean((e_physics - 1 + (mu_1 * (t_physics ** -0.5))) ** 2)
        loss = loss3 + (lambda1 * loss2)
        loss_values.append(loss.item())
        loss3_values.append(loss3.item())
        loss2_values.append(loss2.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
        optimiser.step()

        if i % pinn_range_2 == 0:
            mean_output, std_output = mc_dropout_inference(pinn, t_test_tensor, n_samples=n_mc_samples)
            predictions.append(mean_output.flatten().mean().item())
            uncertainties.append(std_output.flatten().mean().item())


    results_df = pd.DataFrame({
        'Prediction': predictions,
        'Uncertainty': uncertainties
    })

    return results_df
