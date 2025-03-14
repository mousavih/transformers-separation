from setup import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_directory', default='results')

if __name__ == '__main__':
    args = parser.parse_args()
    
    torch.manual_seed(42)
    n_train = 50000
    n_test = 10000

    N = 100
    d = 10
    de = int(np.log(N) * 5)
    
    u = torch.randn(d, 1).to(device)
    u /= torch.norm(u)

    E = 2 * (torch.bernoulli(0.5 * torch.ones(N, de).to(device)) - 0.5) / np.sqrt(de)

    X_train, t_train = generate_input(n_train, N, d, simple=True)
    y_train = ground_truth(X_train, t_train, u)

    X_test, t_test = generate_input(n_test, N, d, simple=True)
    y_test = ground_truth(X_test, t_test, u)
    XE_test = embed_input(X_test, t_test, E)

    trainset = TensorDataset(X_train, t_train, y_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    tr = SimpleTransformer(d, de, 100).to(device)
    train_model_qstr(tr, N, E, trainloader, XE_test, y_test, lr=0.001, verbose=True, weight_decay=0.2)

    attn_proj = tr.attn.in_proj_weight[:d + 2 * de].T @ tr.attn.in_proj_weight[d + 2 * de:2 * (d + 2 * de)]
    attn_proj = attn_proj.detach().cpu()
    
    if not os.path.exists(args.result_directory):
        os.makedirs(args.result_directory)
    
    torch.save(attn_proj, f'{args.result_directory}/attn_proj.pt')
