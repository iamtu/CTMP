Format input files:
    - user:
    each line is a user:
    [#items voted] [id_item_1] [id_item_2] ... [id_item_N]
    - item:
    each line is a item:
    [#users voted] [id_user_1] [id_user_2] ... [id_user_N]
    - mult:
    each line is lda format of each item
    [#words] [word_id:count] [word_id:count] ....

user_id and item_id: 0,1,2...

./ldap --directory=results --user=../data_citeulike/cf-train-1-users.dat --item=../data_citeulike/cf-train-1-items.dat --mult=../data_citeulike/mult.dat --theta_init=../data_citeulike/theta_2.dat --beta_init=../data_citeulike/beta_final.dat

output:
final-muy.dat :item latent factor presenatation
final-eta.dat :user latent factor presenatation

