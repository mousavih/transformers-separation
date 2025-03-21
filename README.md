The code for the paper: "When Do Transformers Outperform Feedforward and Recurrent Networks? A Statistical Perspective".

You can use
```
python sample_complexity.py --model <model_type> --check_every <check_every> --n_test <n_test>
```
where `<model_type>` can be `tr`, `ffn`, or `rnn`, `<check_every>` is the number of online minibatch steps after which the test loss is evaluated, and `<n_test>` is the number of test samples to estimate the test loss. The results will be saved at `./results/` by default. You can then use `plots.ipynb` to generate the sample complexity plots. The plots in the paper are generated by
```
python sample_complexity.py --model <model_type> --check_every 10 --n_test 5000
```
for 1STR and
```
python sample_complexity.py --model <model_type> --simple --check_every 100 --n_test 5000
```
for simple-1STR. To generate the results for the attention weights figure, you can use `python attention_weights.py`. See the code for additional adjustable hyperparameters.
