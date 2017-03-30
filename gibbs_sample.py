import time
#t0 = time.time()
#print time.time()
import pandas as pd
import numpy as nu
import matplotlib as plt
import dp
import math

data = dp.data
label = dp.expert_label
beta = 0.5
alpha = nu.random.rand()
num_st = data.s.unique().shape[0]
num_it = data.e.unique().shape[0]
num_lb = label.ex_lb.unique().shape[0]

tb = pd.DataFrame()
tb_pr = pd.DataFrame()

tb['item'] = pd.Series(data.e.unique())
#intializing the prior of bkt parameter

tb['pr_tb_id'] = pd.Series(-1*nu.ones(num_it))

tb_pr['tb_id'] = pd.Series(nu.r_[range(num_lb)])
tb_pr['bkt_sl'] = pd.Series(nu.random.beta(1,9,num_lb))
tb_pr['bkt_gu'] = pd.Series(nu.random.beta(1,9,num_lb))
tb_pr['bkt_tr'] = pd.Series(nu.random.beta(10,30,num_lb))
tb_pr['bkt_lr'] = pd.Series(nu.random.rand(num_lb))

pr_tb_sl = nu.random.rand(1000)
pr_tb_gu = nu.random.rand(1000)
pr_tb_tr = nu.random.rand(1000)
pr_tb_lr = nu.random.rand(1000)

#likelihood of data if item sit on new table for each prior drwan for BKT parameter



for i in range(num_it):
    tb.loc[i,'pr_tb_id'] = label.iloc[tb.item[i],0]


def gibbs_sample(itr_num):
#crete new parameter column for this iteration
    # print "tb_head",tb[:8]
    if itr_num!=0:
        tb['tb_id'+str(itr_num)] = tb['tb_id'+str(itr_num-1)]
    else:
        tb['tb_id'+str(itr_num)] = tb.pr_tb_id
    column_name = 'tb_id'+str(itr_num)
    ll_seq =0.0

    for item in range(num_it):

#compute sequence probability of table with and withourt item on table
    #removing item from table
            up_tb_rm(item,column_name)

            ll_seq_with_item =[];
            ll_seq_without_item = [];
            log_sitting_pro = [];

            # print tb_pr.head();
            ccc =0;
            for table in tb_pr.tb_id:
                # print table_prior
                # print tb_pr.head();
                up_tb_as(item,column_name,table)
                #log likelihood of table with item
                ll_seq_with_item.append(ll_response(table,column_name))

                up_tb_rm(item,column_name)

                ll_seq_without_item.append(ll_response(table,column_name))

                log_sitting_pro.append(log_sitting_pro_tb(table,column_name,item))
                ccc+=1 ;
                # if(ccc> 4):
                    # break
            #cal probability of item setting at new table
            new_table_id = max(tb_pr.tb_id)+1
            up_tb_as(item,column_name,new_table_id)
            ll_seq_with_item.append(ll_response(new_table_id,column_name))
            ll_seq_without_item.append(0)
            log_sitting_pro.append(nu.log(1-beta)+ nu.log(alpha))
            p_dis = []
            for i in range(len(ll_seq_with_item)):
                p_dis.append(ll_seq_with_item[i]-ll_seq_without_item[i]+log_sitting_pro[i])
            p_dis  = nu.exp(p_dis)
            for i in range(1,len(ll_seq_with_item)):
                p_dis[i] = p_dis[i] + p_dis[i-1]
            partition = p_dis[-1]*nu.random.rand()
            drw = len(ll_seq_with_item)-1
            for i in range(1,len(ll_seq_with_item)):
                if p_dis[i]>=partition:
                    drw = i-1
                    break
            # print "drawn_id",drw
            # print "p_dis ",p_dis
            if drw!=(len(ll_seq_with_item) -1):
                up_tb_rm(item,column_name)
                up_tb_as(item,column_name,tb_pr.tb_id.iloc[drw])
                # print "next_label",tb_pr.tb_id.iloc[drw]
            # print "item" , "prior_tb_id","next_tb_id"
            print item,tb.pr_tb_id[tb.item == item],"next_loc",tb.loc[tb.item== item ,column_name]
            # break
#calculate  probability of itam to sit at table t given all other data
def log_sitting_pro_tb(table,column_name,item_id):
    item = tb.loc[tb.loc[:,column_name] == table, 'item'].values
    item_label_count = label.loc[item,'ex_lb'].value_counts()
    item_label_index = item_label_count.index
    # print item_label_index
    # print 'label'
    # print label.loc[item_id]
    if (item_label_index==label.loc[item_id]).sum():
        k_n = pow((1-beta),-1*item_label_count[label.loc[item_id]])
    else:
        k_n = 1
    k_d = pow ((1-beta),-1*item_label_count.values)
    k_d = k_d.sum()
    k_n_d = k_n/k_d
    st_pr = nu.log(1+beta*(k_n_d-1))+nu.log(item_label_count.values.sum()) - nu.log(1+beta*(1/num_lb-1))
    # st_pr = nu.log(1+beta*(k_n_d-1))
    # st_pr =0.0
    return st_pr
    print k_d


#calculate student response likelihood for particular table

def ll_response(tb_id,column_name):
    item = tb.loc[tb.loc[:,column_name]== tb_id , 'item'].values;
    data_temp = data.loc[data.e.isin(item)]
    # print data_temp.s.unique()
    # print 'student'
    parameter = tb_pr.loc[tb_pr.tb_id == tb_id].values
    parameter= parameter[0,1:]
    p_slip  = parameter[0]
    p_guess = parameter[1]
    p_transition = parameter[2]
    p_learned = parameter[3]
    ll_rs_seq = 0.0
    # print parameter

    for student in data_temp.s.unique():
        K_trans_prob = p_learned
        response= data_temp.loc[data_temp.s==student,'r'].values
        # print response.values.shape
        # first_response == True
        for res in response:
            if(res):
                ll_rs_seq += nu.log(K_trans_prob*(1-p_slip)+(1-K_trans_prob)*p_guess)
                K_trans_prob = (K_trans_prob*(1-p_slip)+p_transition*(1-K_trans_prob)*p_guess)/(K_trans_prob*(1-p_slip)+(1-K_trans_prob)*p_guess)
            else:
                ll_rs_seq += (1- (K_trans_prob*(1-p_slip)+(1-K_trans_prob)*p_guess))
                K_trans_prob = (K_trans_prob*p_slip + p_transition*(1-K_trans_prob)*(1-p_guess))/(K_trans_prob*p_slip+(1-K_trans_prob)*(1-p_guess))
            # print ll_rs_seq,K_trans_prob
        # break
    # print data_temp
    # if ll_rs_seq > 0:
        # print parameter
    return ll_rs_seq
    # return 0

#remove parameter for given table id
def rm_tb_pr(tb_id):
    tb_pr.drop(tb_pr.index[tb_pr.tb_id == tb_id],inplace = True)

def up_tb_as(item ,column_name,tb_id):
    tb.loc[tb.item == item ,column_name] = tb_id
    # print tb_id
    # print "tb_id"
    #check if table exist if not create new table and sample new parameter
    # print tb_pr.head()
    if ((tb_pr.tb_id == tb_id).sum())==0:
        temp_index = tb_pr.index.max()+1
        # print temp_index
        # print nu.hstack((nu.r_[tb_id],nu.random.rand(4)))
        # tb_pr[temp_index]
        tb_pr.loc[temp_index] = nu.hstack((nu.r_[tb_id],nu.r_[nu.random.beta(1,9),nu.random.beta(1,9),nu.random.beta(10,30),nu.random.rand()]))

def up_tb_rm(item,column_name):
    table_id = tb.loc[tb.item == item,column_name].values[0]
    # print table_id
    if (tb.loc[:,column_name]== table_id).sum() == 1:
        rm_tb_pr(table_id)
    tb.loc[tb.item == item ,column_name] = -1


gibbs_sample(0);
tb.to_csv('item_table_assignmnet.csv')
tb_pr.to_csv('table_parameter.csv')
