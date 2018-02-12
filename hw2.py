# IEMS 308
# HW 2 - Association Rules
# Alex Merryman

import psycopg2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

login_dict = {"hostname":"gallery.iems.northwestern.edu", "database_name":"iems308",
              "db_port":"5432", "username":"afm900", "pw":"afm900_pw"}


conn = psycopg2.connect(host=login_dict["hostname"], database=login_dict["database_name"],
                        user=login_dict["username"], port=login_dict["db_port"],
                        password=login_dict["pw"])

cur = conn.cursor()


# SAMPLING STRATEGY
def drop_table():
    '''Drops the afm900_schema.trnsact if it exists. Used to ensure the schema
    is clear before creating the table and inserting data.'''

    drop_table_command = "DROP TABLE IF EXISTS afm900_schema.trnsact"
    cur.execute(drop_table_command)
    print 'Table dropped (if exists)'


def create_table():
    '''Creates the table afm900_schema.trnsact'''

    create_trnsact_table_command = """CREATE TABLE afm900_schema.trnsact (
                                        sku        varchar(80),
                                        store      varchar(80),
                                        register   varchar(80),
                                        trannum    varchar(80),
                                        seq        varchar(80),
                                        saledate   varchar(80),
                                        stype      varchar(80),
                                        qty        varchar(80),
                                        orgprice   varchar(80),
                                        sprice     varchar(80),
                                        c11        varchar(80),
                                        interid    varchar(80),
                                        mic        varchar(80),
                                        c14        varchar(80)
                                       );"""



    cur.execute(create_trnsact_table_command)
    print 'Table created'


def sample_trnsact(date_str):
    '''Inserts a specific date's transaction data into afm900_schema.trnsact,
    then drops the c11 and c14 columns.'''

    drop_table()
    create_table()

    insert_command = "INSERT INTO afm900_schema.trnsact SELECT * FROM pos.trnsact WHERE c6='{}';".format(date_str)
    cur.execute(insert_command)

    # c11, c14 redundant/useless
    drop_cols_command = "ALTER TABLE afm900_schema.trnsact DROP COLUMN c11, DROP COLUMN c14;"
    cur.execute(drop_cols_command)

    cur.close()
    conn.commit()


sample_trnsact('2005-07-12')

# TRANSACTION EXPLORATION
trnsact_df_query = "SELECT * FROM afm900_schema.trnsact;"
trnsact_df = pd.read_sql_query(trnsact_df_query, conn)

print '-----------------------------'
print 'Query Successful'
print '-----------------------------'


# DATA TYPE CONVERSION
trnsact_df_type_dict = {'sku':str, 'store':str, 'register':str, 'trannum':str,
                        'seq':str, 'stype':str, 'qty':np.int,
                        'orgprice':np.float, 'sprice':np.float, 'interid':str, 'mic':str}

# convert saledate column into datetime format
trnsact_df['saledate'] = pd.to_datetime(trnsact_df['saledate'])

# convert other date in dateframe to dtypes from dictionary
trnsact_df = trnsact_df.astype(dtype=trnsact_df_type_dict)


# pickle the transaction dataframe for easy reference
current_wd = os.getcwd()
pickle_filename = os.path.join(current_wd, 'trnsaction_df_pickle.pkl')
trnsact_df.to_pickle(pickle_filename)

trnsact_df = pd.read_pickle(pickle_filename)

print 'Data conversion Successful'
print '-----------------------------'


# SKU STATS
# number of unique SKU values
unique_SKUs = trnsact_df.sku.unique()
print '# unique SKUs:', len(unique_SKUs)

# unique SKU value counts
unique_SKU_counts = trnsact_df.sku.value_counts()

# STORE STATS
# number of unique stores
unique_stores = trnsact_df.store.unique()
print '# unique stores:', len(unique_stores)

# unique store counts
unique_store_counts = trnsact_df.store.value_counts()
highest_volume_stores = trnsact_df['store'].value_counts().index.tolist()


# REGISTER STATS
# number of unique registers
unique_registers = trnsact_df.register.unique()
print '# unique registers:', len(unique_registers)

# unique register counts
unique_register_counts = trnsact_df.register.value_counts()

# TRANNUM STATS
# number of unique trannum values
unique_trannum = trnsact_df.trannum.unique()
print '# unique trannum:', len(unique_trannum)
print '-----------------------------'


## Plot histogram of SKUs appearing over 100 times
#plot_SKU_counts_df = unique_SKU_counts.drop(unique_SKU_counts[unique_SKU_counts < 100].index)
#
#SKU_hist = sns.distplot(plot_SKU_counts_df, kde=False)
#fig = SKU_hist.get_figure()
#fig.savefig('SKU_histogram.png')
#
#
## Plot histogram of stores appearing over 1000 times
#plot_store_counts_df = unique_store_counts.drop(unique_store_counts[unique_store_counts < 1000].index)
#
#store_hist = sns.distplot(plot_store_counts_df, kde=False)
#fig2 = store_hist.get_figure()
#fig2.savefig('store_histogram.png')


# filter trnsact_df to include only transactions that occured at the 20 highest volume stores
HV_stores = highest_volume_stores[0:20]
HV_stores_trnsact = trnsact_df[trnsact_df['store'].isin(HV_stores)]

u_stores_HV = HV_stores_trnsact.store.unique()
u_registers_HV = HV_stores_trnsact.register.unique()
u_trannum_HV = HV_stores_trnsact.trannum.unique()


list_of_baskets = []

store_register_trannum_grouped = HV_stores_trnsact.groupby(['store', 'register', 'trannum'])

for name, group in store_register_trannum_grouped:
    (u_s, u_r, u_t) = name
    basket_SKUs = group['sku'].unique().tolist()
    if len(basket_SKUs) > 1:
        d = {'store':u_s, 'register':u_r, 'trannum':u_t, 'basket':basket_SKUs}
        list_of_baskets.append(d)
    else:
        pass


basket_df = pd.DataFrame(list_of_baskets)
all_baskets = basket_df['basket'].tolist()


from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

oht = OnehotTransactions()
oht_array = oht.fit(all_baskets).transform(all_baskets)

oht_df = pd.DataFrame(oht_array, columns=oht.columns_)
print 'One Hot Encoding Successful'
print '-----------------------------'

frequent_SKUs = apriori(oht_df, min_support=0.0004, use_colnames=True)
print 'Frequent SKUs'


print '----------------------------------------------------------'

assoc_rules = association_rules(frequent_SKUs, metric="confidence", min_threshold=0.5)

# pickle the association rules dataframe for easy reference
current_wd = os.getcwd()
assoc_rule_pickle_filename = os.path.join(current_wd, 'assoc_rule_pickle.pkl')
assoc_rules.to_pickle(assoc_rule_pickle_filename)

assoc_rules = pd.read_pickle(assoc_rule_pickle_filename)

rules_sorted = assoc_rules.sort_values(by=['lift'], ascending=False)


def combine_antecedants_consequents(row):
    antecedant_list = list(row['antecedants'])
    consequent_list = list(row['consequents'])
    combined = [antecedant_list + consequent_list]
    return sorted(combined[0])


rules_sorted['a+c'] = rules_sorted.apply(combine_antecedants_consequents, axis=1)
rules_sorted.to_csv('assoc_rules.csv')


def num_unique_SKUs(sub_df):
    SKU_list = sub_df['a+c']
    flat_list = [item for sublist in SKU_list for item in sublist]
    rules_unique_SKUs = set(flat_list)
    num_SKUs = len(rules_unique_SKUs)
    return rules_unique_SKUs, num_SKUs


# Select top 173382 rules by lift to find 100+ unique SKUs as candidates to move
rules_sorted_sub_df = rules_sorted[0:173382]
rules_unique_SKUs, num_SKUs = num_unique_SKUs(rules_sorted_sub_df)
print 'Top 100+ Candidate SKUs for Rearrangement'
print rules_unique_SKUs

