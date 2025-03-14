from setup import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['ffn', 'rnn', 'tr'])
parser.add_argument('--simple', action='store_true')
parser.add_argument('--check_every', type=int, default=200)
parser.add_argument('--n_test', type=int, default=5000)
parser.add_argument('--loss_thrsh', type=float)
parser.add_argument('--result_directory', default='results')

if __name__=='__main__':    
    args = parser.parse_args()

    d = 10
    n_test = args.n_test
    test_batch_size=1000
    
    torch.manual_seed(42)
    samples_all = []

    u = torch.randn(d, 1).to(device)
    u /= torch.norm(u)
    
    if args.simple:
        N_range = range(25, 35, 2)
    elif args.model == 'ffn':
        N_range = range(2, 13, 2)
    else:
        N_range = range(2, 13, 2)

    Es = {}
    for N in N_range:
        de = int(np.log(N) * 5)
        Es[N] = 2 * (torch.bernoulli(0.5 * torch.ones(N, de).to(device)) - 0.5) / np.sqrt(de)

    for _ in range(5):
        samples = []
        for N in N_range:
            print(f'N: {N}')
            de = int(np.log(N) * 5)
            
            
            if args.model == 'ffn':
                model = FFN(d + 2 * de, N, [N * d, 1000]).to(device)
            if args.model == 'rnn':
                model = BGRU(d, de, h_size=500, bidirectional=True).to(device)
            if args.model == 'tr':
                model = SimpleTransformer(d, de, 100).to(device)
            
            
            X_test, t_test = generate_input(n_test, N, d, simple=True)
            y_test = ground_truth(X_test, t_test, u)
            XE_test = embed_input(X_test, t_test, Es[N])

            testset = TensorDataset(XE_test, y_test)
            testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
            
            samples.append(find_sample_complexity(N, d, de, model, u, Es[N], testloader, loss_thr=args.loss_thrsh, lr=0.001, check_every=args.check_every, simple=args.simple))
            print()
        samples_all.append(samples)
        
    if not os.path.exists(args.result_directory):
        os.makedirs(args.result_directory)
    
    if args.simple:
        if args.model == 'ffn':
            torch.save(samples_all, f'{args.result_directory}/ffn_simple_{args.check_every}_{n_test}.pt')
        if args.model == 'rnn':
            torch.save(samples_all, f'{args.result_directory}/rnn_simple_{args.check_every}_{n_test}.pt')
        if args.model == 'tr':
            torch.save(samples_all, f'{args.result_directory}/tr_simple_{args.check_every}_{n_test}.pt')
    else:
        if args.model == 'ffn':
            torch.save(samples_all, f'{args.result_directory}/ffn_{args.check_every}_{n_test}.pt')
        if args.model == 'rnn':
            torch.save(samples_all, f'{args.result_directory}/rnn_{args.check_every}_{n_test}.pt')
        if args.model == 'tr':
            torch.save(samples_all, f'{args.result_directory}/tr_{args.check_every}_{n_test}.pt')
        