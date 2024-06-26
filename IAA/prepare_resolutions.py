import ast

def get_checked_annotations(df):
    tuples_classes = []

    # loop over (concatenated) df
    for index, row in df.iterrows():
        # if the annotation only spans one token
        if len(ast.literal_eval(row['mention_ids'])) == 1:
            # and if the resolution is not to not annotate
            if row['Should this mention be annotated as an event?'] == 'yes':
                # then append the token_id and resolved class to the list of tuples
                tuples_classes.append((ast.literal_eval(row['mention_ids'])[0], row['If so, which event class should the mention be annotated with?']))
        # if the annotation spans more than one token
        if len(ast.literal_eval(row['mention_ids'])) > 1:
            # and if the resolution is not to not annotate
            if row['Should this mention be annotated as an event?'] == 'yes':
                # create a tuple for each token
                for i in range(0, len(ast.literal_eval(row['mention_ids']))):
                    tuples_classes.append((ast.literal_eval(row['mention_ids'])[i], row['If so, which event class should the mention be annotated with?']))


    return(tuples_classes)

def get_resolutions(df):

    tuples_classes = []
    tuples_anchors = []

    #loop over (concatenated) df
    for index, row in df.iterrows():
        # if the annotation only spans one token
        if len(ast.literal_eval(row['Resolved span'])) == 1:
            # if the resolution was to not annotate:
            if row['Resolved class'] == '/':
                tuples_classes.append((ast.literal_eval(row['Resolved span'])[0], '0'))
                tuples_anchors.append((ast.literal_eval(row['Resolved span'])[0], '0'))
            # if the resolution did consist of an annotation
            if row['Resolved class'] != '/':
                # then append the token_id and resolved class to the list of tuples
                tuples_classes.append((ast.literal_eval(row['Resolved span'])[0], row['Resolved class']))
                tuples_anchors.append((ast.literal_eval(row['Resolved span'])[0], row['anchor_type']))
        # if the annotation spans more than one token
        if len(ast.literal_eval(row['Resolved span'])) > 1:
            # and if the resolution is not to not annotate
            if row['Resolved class'] == '/':
                for i in range(0, len(ast.literal_eval(row['Resolved span']))):
                    tuples_classes.append((ast.literal_eval(row['Resolved span'])[i], '0'))
                    tuples_anchors.append((ast.literal_eval(row['Resolved span'])[i], '0'))
            if row['Resolved class'] != '/':
                # create a tuple for each token
                for i in range(0, len(ast.literal_eval(row['Resolved span']))):
                    tuples_classes.append((ast.literal_eval(row['Resolved span'])[i], row['Resolved class']))
                    tuples_anchors.append((ast.literal_eval(row['Resolved span'])[i], row['anchor_type']))

    return(tuples_classes, tuples_anchors)



