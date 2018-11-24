mkdir results
echo '======= p0.01 ======='
time ./ldap --directory=results/p0.01 --p=0.01 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.01

echo '======= p0.1 ======='
time ./ldap --directory=results/p0.1 --p=0.1 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.1

echo '======= p0.2 ======='
time ./ldap --directory=results/p0.2 --p=0.2 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.2

echo '======= p0.3 ======='
time ./ldap --directory=results/p0.3 --p=0.3 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.3

echo '======= p0.4 ======='
time ./ldap --directory=results/p0.4 --p=0.4 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.4

echo '======= p0.5 ======='
time ./ldap --directory=results/p0.5 --p=0.5 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.5

echo '======= p0.6 ======='
time ./ldap --directory=results/p0.6 --p=0.6 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.6

echo '======= p0.7 ======='
time ./ldap --directory=results/p0.7 --p=0.7 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.7

echo '======= p0.8 ======='
time ./ldap --directory=results/p0.8 --p=0.8 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.8


echo '======= p0.9 ======='
time ./ldap --directory=results/p0.9 --p=0.9 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.9

echo '======= p0.99 ======='
time ./ldap --directory=results/p0.99 --p=0.99 --alpha=0.01 --user=../data_citeulike/users_train.dat --item=../data_citeulike/items_train.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat --n_thread=20 --max_iter=100 --save_lag=20
python eval.py p0.99



