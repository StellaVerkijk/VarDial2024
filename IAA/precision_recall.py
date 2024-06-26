import pandas as pd
from prettytable import PrettyTable
from prepare_resolutions import get_checked_annotations

B = pd.read_csv('../annotated_data/IAA/Round1/tsv/Brecht-annotations.tsv')
M = pd.read_csv('../annotated_data/IAA/Round1/tsv/Manjusha-annotations.tsv')
L = pd.read_csv('../annotated_data/IAA/Round1/tsv/Lodewijk-annotations.tsv')
K = pd.read_csv('../annotated_data/IAA/Round1/tsv/Kay-annotations.tsv')

def drop_splitup_tokens(df):
    indices_to_drop = []
    for index, row in df.iterrows():
        if '.' in row['token_id']:
            indices_to_drop.append(index)
    new_df = df.drop(indices_to_drop)
    return(new_df)

M = drop_splitup_tokens(M)
K = drop_splitup_tokens(K)
B = drop_splitup_tokens(B)
L = drop_splitup_tokens(L)

annos_kay = K['eventclass_no_number'].tolist()
annos_br = B['eventclass_no_number'].tolist()
annos_manj = M['eventclass_no_number'].tolist()
annos_lod = L['eventclass_no_number'].tolist()

def delete_mistakes(l):
    new = []
    for item in l:
        if item == '*':
            new.append('0')
        elif item == 'nan':
            new.append('0')
        else:
            new.append(item)

    return(new)

annos_kay = delete_mistakes(annos_kay)
annos_br = delete_mistakes(annos_br)
annos_manj = delete_mistakes(annos_manj)
annos_lod = delete_mistakes(annos_lod)

print('Calculating Precision and Recall comparing annotators to test set')

def calculate(zipped_list):
    tp = 0
    fp = 0
    fn = 0

    for item in zipped_list:
        if item[0] == '0' and item[1] != '0':
            fn += 1
        if item[0] != '0' and item[1] == '0':
            fp += 1
        if item[0] != '0' and item[1] != '0':
            tp += 1

    precision = tp / (tp +fp)
    recall = tp / (tp +fn)
    number_annos = tp+fp

    return(precision, recall, number_annos)


# Test set as gold
gold = pd.read_csv('../annotated_data/curated_test/FINAL_testset.tsv', sep='\t')
gold_annos = gold['final_event'].tolist()
print('asserting same lengths...')
assert(len(gold_annos)==len(annos_kay)==len(annos_lod)==len(annos_br)==len(annos_manj))
print()
print('..................................................................')
print()
print('Calculating Precision and Recall for event detection comparing annotators to the test set')


#Comparing to annotations before check task
print()
#Manjusha (Ann1)
zipped = zip(annos_manj, gold_annos)
zipped_list = list(zipped)
precision_1, recall_1, n_1 = calculate(zipped_list)

#Kay (Ann2)
zipped = zip(annos_kay, gold_annos)
zipped_list = list(zipped)
precision_2, recall_2, n_2 = calculate(zipped_list)

#Brecht (Ann3)
zipped = zip(annos_br, gold_annos)
zipped_list = list(zipped)
precision_3, recall_3, n_3 = calculate(zipped_list)

#Lodewijk (Ann4)
zipped = zip(annos_lod, gold_annos)
zipped_list = list(zipped)
precision_4, recall_4, n_4 = calculate(zipped_list)


print("Comparing gold to annotations before check task")
table = PrettyTable()
table.field_names = ["", "Precision", "Recall", "n"]
table.add_row(["Ann1", precision_1, recall_1, n_1])
table.add_row(["Ann2", precision_2, recall_2, n_2])
table.add_row(["Ann3", precision_3, recall_3, n_3])
table.add_row(["Ann4", precision_4, recall_4, n_4])
print(table)


def insert_annos(df, df_checks):

    tuples = get_checked_annotations(df_checks)

    i = -1
    token_ids = df['token_id'].tolist()
    indices = []
    for item in token_ids:
        i += 1
        for x in tuples:
            if x[0] == item:
                indices.append((i, x[1]))

    classes = df['eventclass_no_number'].tolist()
    for x in indices:
        classes[x[0]] = x[1]

    df['eventclass_no_number'] = classes
    return(df)

#checks Manjusha
Mcheck2 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Mcheck2.tsv', delimiter='\t')
Mcheck3 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Mcheck3.tsv', delimiter='\t')
Mcheck4 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Mcheck4.tsv', delimiter='\t')
Mcheck = pd.concat([Mcheck2, Mcheck3, Mcheck4])

M = insert_annos(M, Mcheck)

#checks Kay
Kcheck1 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Kcheck1.tsv', delimiter='\t')
Kcheck3 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Kcheck3.tsv', delimiter='\t')
Kcheck4 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Kcheck4.tsv', delimiter='\t')
Kcheck = pd.concat([Kcheck1, Kcheck3, Kcheck4])

K = insert_annos(K, Kcheck)

#checks Brecht
Bcheck1 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Bcheck1.tsv', delimiter='\t')
Bcheck2 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Bcheck2.tsv', delimiter='\t')
Bcheck4 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Bcheck4.tsv', delimiter='\t')
Bcheck = pd.concat([Bcheck1, Bcheck2, Bcheck4])

B = insert_annos(B, Bcheck)

#checks Lodewijk
Lcheck1 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Lcheck1.tsv', delimiter='\t')
Lcheck2 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Lcheck2.tsv', delimiter='\t')
Lcheck3 = pd.read_csv('../annotated_data/IAA/Round1/checked_data/Lcheck3.tsv', delimiter='\t')
Lcheck = pd.concat([Lcheck1, Lcheck2, Lcheck3])

L = insert_annos(L, Lcheck)


annos_kay = K['eventclass_no_number'].tolist()
annos_br = B['eventclass_no_number'].tolist()
annos_manj = M['eventclass_no_number'].tolist()
annos_lod = L['eventclass_no_number'].tolist()


print('asserting same lengths...')
assert(len(gold_annos)==len(annos_kay)==len(annos_lod)==len(annos_br)==len(annos_manj))
print()

#Comparing to annotations after check task

#Manjusha (Ann1)
zipped = zip(annos_manj, gold_annos)
zipped_list = list(zipped)
precision_1, recall_1, n_1 = calculate(zipped_list)

#Kay (Ann2)
zipped = zip(annos_kay, gold_annos)
zipped_list = list(zipped)
precision_2, recall_2, n_2 = calculate(zipped_list)

#Brecht (Ann3)
zipped = zip(annos_br, gold_annos)
zipped_list = list(zipped)
precision_3, recall_3, n_3 = calculate(zipped_list)

#Lodewijk (Ann4)
zipped = zip(annos_lod, gold_annos)
zipped_list = list(zipped)
precision_4, recall_4, n_4 = calculate(zipped_list)


print("Comparing gold to annotations after check task ")
table = PrettyTable()
table.field_names = ["", "Precision", "Recall", "n"]
table.add_row(["Ann1", precision_1, recall_1, n_1])
table.add_row(["Ann2", precision_2, recall_2, n_2])
table.add_row(["Ann3", precision_3, recall_3, n_3])
table.add_row(["Ann4", precision_4, recall_4, n_4])
print(table)


print()
print('..................................................................')
