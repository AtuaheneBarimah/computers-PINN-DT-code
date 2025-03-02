# @title PINN Models of Components
class ModelFCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, dropout_rate=dropout_rate_x):
        super().__init__()
        activation = nn.LeakyReLU
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation(),
            nn.Dropout(dropout_rate)
        )
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation(),
                nn.Dropout(dropout_rate)
            ) for _ in range(N_LAYERS - 1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
def mc_dropout_inference(pinn, t_test_tensor, n_samples=100):
    pinn.train()
    outputs = []

    for _ in range(n_samples):
        outputs.append(pinn(t_test_tensor).detach().numpy())

    outputs = np.array(outputs)
    mean_output = np.mean(outputs, axis=0)
    std_output = np.std(outputs, axis=0)

    return mean_output, std_output
