mkdir results
echo 'p0.01'
time ./ldap --directory=results/p0.01 --p=0.01 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.01
