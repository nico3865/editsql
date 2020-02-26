"""Basic model training and evaluation functions."""

from enum import Enum

import random
import sys
import json
import progressbar
import model.torch_utils
import data_util.sql_util
import torch

def write_prediction(fileptr,
                     identifier,
                     input_seq,
                     probability,
                     prediction,
                     flat_prediction,
                     gold_query,
                     flat_gold_queries,
                     gold_tables,
                     index_in_interaction,
                     database_username,
                     database_password,
                     database_timeout,
                     compute_metrics=True):
    pred_obj = {}
    pred_obj["identifier"] = identifier
    if len(identifier.split('/')) == 2:
        database_id, interaction_id = identifier.split('/')
    else:
        database_id = 'atis'
        interaction_id = identifier
    pred_obj["database_id"] = database_id
    pred_obj["interaction_id"] = interaction_id

    pred_obj["input_seq"] = input_seq
    pred_obj["probability"] = probability
    pred_obj["prediction"] = prediction
    pred_obj["flat_prediction"] = flat_prediction
    pred_obj["gold_query"] = gold_query
    pred_obj["flat_gold_queries"] = flat_gold_queries
    pred_obj["index_in_interaction"] = index_in_interaction
    pred_obj["gold_tables"] = str(gold_tables)

    # Now compute the metrics we want.

    if compute_metrics:
        # First metric: whether flat predicted query is in the gold query set.
        correct_string = " ".join(flat_prediction) in [
            " ".join(q) for q in flat_gold_queries]
        pred_obj["correct_string"] = correct_string

        # Database metrics
        if not correct_string:
            syntactic, semantic, pred_table = sql_util.execution_results(
                " ".join(flat_prediction), database_username, database_password, database_timeout)
            pred_table = sorted(pred_table)
            best_prec = 0.
            best_rec = 0.
            best_f1 = 0.

            for gold_table in gold_tables:
                num_overlap = float(len(set(pred_table) & set(gold_table)))

                if len(set(gold_table)) > 0:
                    prec = num_overlap / len(set(gold_table))
                else:
                    prec = 1.

                if len(set(pred_table)) > 0:
                    rec = num_overlap / len(set(pred_table))
                else:
                    rec = 1.

                if prec > 0. and rec > 0.:
                    f1 = (2 * (prec * rec)) / (prec + rec)
                else:
                    f1 = 1.

                best_prec = max(best_prec, prec)
                best_rec = max(best_rec, rec)
                best_f1 = max(best_f1, f1)

        else:
            syntactic = True
            semantic = True
            pred_table = []
            best_prec = 1.
            best_rec = 1.
            best_f1 = 1.

        assert best_prec <= 1.
        assert best_rec <= 1.
        assert best_f1 <= 1.
        pred_obj["syntactic"] = syntactic
        pred_obj["semantic"] = semantic
        correct_table = (pred_table in gold_tables) or correct_string
        pred_obj["correct_table"] = correct_table
        pred_obj["strict_correct_table"] = correct_table and syntactic
        pred_obj["pred_table"] = str(pred_table)
        pred_obj["table_prec"] = best_prec
        pred_obj["table_rec"] = best_rec
        pred_obj["table_f1"] = best_f1

    fileptr.write(json.dumps(pred_obj) + "\n")

class Metrics(Enum):
    """Definitions of simple metrics to compute."""
    LOSS = 1
    TOKEN_ACCURACY = 2
    STRING_ACCURACY = 3
    CORRECT_TABLES = 4
    STRICT_CORRECT_TABLES = 5
    SEMANTIC_QUERIES = 6
    SYNTACTIC_QUERIES = 7


def get_progressbar(name, size):
    """Gets a progress bar object given a name and the total size.

    Inputs:
        name (str): The name to display on the side.
        size (int): The maximum size of the progress bar.

    """
    return progressbar.ProgressBar(maxval=size,
                                   widgets=[name,
                                            progressbar.Bar('=', '[', ']'),
                                            ' ',
                                            progressbar.Percentage(),
                                            ' ',
                                            progressbar.ETA()])


def train_epoch_with_utterances(batches,
                                model,
                                randomize=True):
    """Trains model for a single epoch given batches of utterance data.

    Inputs:
        batches (UtteranceBatch): The batches to give to training.
        model (ATISModel): The model obect.
        learning_rate (float): The learning rate to use during training.
        dropout_amount (float): Amount of dropout to set in the model.
        randomize (bool): Whether or not to randomize the order that the batches are seen.
    """
    if randomize:
        random.shuffle(batches)
    progbar = get_progressbar("train     ", len(batches))
    progbar.start()
    loss_sum = 0.

    for i, batch in enumerate(batches):
        batch_loss = model.train_step(batch)
        loss_sum += batch_loss

        progbar.update(i)

    progbar.finish()

    total_loss = loss_sum / len(batches)

    return total_loss


def train_epoch_with_interactions(interaction_batches,
                                  params,
                                  model,
                                  randomize=True):
    """Trains model for single epoch given batches of interactions.

    Inputs:
        interaction_batches (list of InteractionBatch): The batches to train on.
        params (namespace): Parameters to run with.
        model (ATISModel): Model to train.
        randomize (bool): Whether or not to randomize the order that batches are seen.
    """
    if randomize:
        random.shuffle(interaction_batches)
    progbar = get_progressbar("train     ", len(interaction_batches))
    progbar.start()
    loss_sum = 0.

    for i, interaction_batch in enumerate(interaction_batches):
        assert len(interaction_batch) == 1
        interaction = interaction_batch.items[0]

        if interaction.identifier == "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5":
            continue

        if 'sparc' in params.data_directory and "baseball_1" in interaction.identifier:
            continue

        batch_loss = model.train_step(interaction, params.train_maximum_sql_length)

        loss_sum += batch_loss
        torch.cuda.empty_cache()

        progbar.update(i)

    progbar.finish()

    total_loss = loss_sum / len(interaction_batches)

    return total_loss


def update_sums(metrics,
                metrics_sums,
                predicted_sequence,
                flat_sequence,
                gold_query,
                original_gold_query,
                gold_forcing=False,
                loss=None,
                token_accuracy=0.,
                database_username="",
                database_password="",
                database_timeout=0,
                gold_table=None):
    """" Updates summing for metrics in an aggregator.

    TODO: don't use sums, just keep the raw value.
    """
    if Metrics.LOSS in metrics:
        metrics_sums[Metrics.LOSS] += loss.item()
    if Metrics.TOKEN_ACCURACY in metrics:
        if gold_forcing:
            metrics_sums[Metrics.TOKEN_ACCURACY] += token_accuracy
        else:
            num_tokens_correct = 0.
            for j, token in enumerate(gold_query):
                if len(
                        predicted_sequence) > j and predicted_sequence[j] == token:
                    num_tokens_correct += 1
            metrics_sums[Metrics.TOKEN_ACCURACY] += num_tokens_correct / \
                len(gold_query)
    if Metrics.STRING_ACCURACY in metrics:
        metrics_sums[Metrics.STRING_ACCURACY] += int(
            flat_sequence == original_gold_query)

    if Metrics.CORRECT_TABLES in metrics:
        assert database_username, "You did not provide a database username"
        assert database_password, "You did not provide a database password"
        assert database_timeout > 0, "Database timeout is 0 seconds"

        # Evaluate SQL
        if flat_sequence != original_gold_query:
            syntactic, semantic, table = sql_util.execution_results(
                " ".join(flat_sequence), database_username, database_password, database_timeout)
        else:
            syntactic = True
            semantic = True
            table = gold_table

        metrics_sums[Metrics.CORRECT_TABLES] += int(table == gold_table)
        if Metrics.SYNTACTIC_QUERIES in metrics:
            metrics_sums[Metrics.SYNTACTIC_QUERIES] += int(syntactic)
        if Metrics.SEMANTIC_QUERIES in metrics:
            metrics_sums[Metrics.SEMANTIC_QUERIES] += int(semantic)
        if Metrics.STRICT_CORRECT_TABLES in metrics:
            metrics_sums[Metrics.STRICT_CORRECT_TABLES] += int(
                table == gold_table and syntactic)


def construct_averages(metrics_sums, total_num):
    """ Computes the averages for metrics.

    Inputs:
        metrics_sums (dict Metric -> float): Sums for a metric.
        total_num (int): Number to divide by (average).
    """
    metrics_averages = {}
    for metric, value in metrics_sums.items():
        metrics_averages[metric] = value / total_num
        if metric != "loss":
            metrics_averages[metric] *= 100.

    return metrics_averages


def evaluate_utterance_sample(sample,
                              model,
                              max_generation_length,
                              name="",
                              gold_forcing=False,
                              metrics=None,
                              total_num=-1,
                              database_username="",
                              database_password="",
                              database_timeout=0,
                              write_results=False):
    """Evaluates a sample of utterance examples.

    Inputs:
        sample (list of Utterance): Examples to evaluate.
        model (ATISModel): Model to predict with.
        max_generation_length (int): Maximum length to generate.
        name (str): Name to log with.
        gold_forcing (bool): Whether to force the gold tokens during decoding.
        metrics (list of Metric): Metrics to evaluate with.
        total_num (int): Number to divide by when reporting results.
        database_username (str): Username to use for executing queries.
        database_password (str): Password to use when executing queries.
        database_timeout (float): Timeout on queries when executing.
        write_results (bool): Whether to write the results to a file.
    """
    assert metrics

    if total_num < 0:
        total_num = len(sample)

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with filename " + str(name) + "_predictions.json")
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    predictions = []
    for i, item in enumerate(sample):
        _, loss, predicted_seq = model.eval_step(
            item, max_generation_length, feed_gold_query=gold_forcing)
        loss = loss / len(item.gold_query())
        predictions.append(predicted_seq)

        flat_sequence = item.flatten_sequence(predicted_seq)
        token_accuracy = torch_utils.per_token_accuracy(
            item.gold_query(), predicted_seq)

        if write_results:
            write_prediction(
                predictions_file,
                identifier=item.interaction.identifier,
                input_seq=item.input_sequence(),
                probability=0,
                prediction=predicted_seq,
                flat_prediction=flat_sequence,
                gold_query=item.gold_query(),
                flat_gold_queries=item.original_gold_queries(),
                gold_tables=item.gold_tables(),
                index_in_interaction=item.utterance_index,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout)

        update_sums(metrics,
                    metrics_sums,
                    predicted_seq,
                    flat_sequence,
                    item.gold_query(),
                    item.original_gold_queries()[0],
                    gold_forcing,
                    loss,
                    token_accuracy,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=item.gold_tables()[0])

        progbar.update(i)

    progbar.finish()
    predictions_file.close()

    return construct_averages(metrics_sums, total_num), None






def evaluate_interaction_sample_NIC_2(sample,
                                model,
                                max_generation_length,
                                name="",
                                gold_forcing=False,
                                metrics=None,
                                total_num=-1,
                                database_username="",
                                database_password="",
                                database_timeout=0,
                                use_predicted_queries=False,
                                write_results=False,
                                use_gpu=False,
                                compute_metrics=False):

    print(" INSIDE --> evaluate_interaction_sample_NIC_2")
    num_utterances = 0
    predictions = []
    model.eval()
    for i, interaction_ in enumerate(sample):
        interaction = interaction_
        break
    print("------------")
    print("interaction")
    print("------------")

    list_of_all_utt = [
        # ['Show', 'the', 'template', 'id', 'for', 'all', 'documents', '.'],
        # ['Show', 'all', 'distinct', 'results', '.'],
        # ['How', 'many', 'are', 'there', '?']
        'Show the template id for all documents .',
        'Show all distinct results .',
        'How many are there ?'
    ]
    list_of_previous_utt = []
    counter = -1
    for utt in list_of_all_utt:
        
        counter += 1

        # create utterance
        #import data_util.utterance
        example_utt_CLONE = data_util.utterance.Utterance(
                utt, #example,
                [], #available_snippets,
                {}, #nl_to_sql_dict,
                params,
                anon_tok_to_ent,
                anonymizer)
        # ((((((((((((((((((((((((((((((~~~~~~~~~~~~~~~~~~~~))))))))))))))))))))))))))))))
        # vars(interaction.interaction.utterances[0])
        # { 'original_input_seq': ['Show', 'information', 'for', 'all', 'documents', '.'], 
        #   'available_snippets': [], 
        #   'keep': True, 
        #   'input_seq_to_use': ['Show', 'information', 'for', 'all', 'documents', '.'], 
        #   'original_gold_query': ['select', 'documents.*'], 
        #   'gold_sql_results': [], 
        #   'contained_entities': [
        #       ['DISTINCT', 'flight.flight_id', ',', 'flight.flight_days', ',', 'flight.from_airport', ',', 'flight.to_airport', ',', 'flight.departure_time', ',', 'flight.arrival_time', ',', 'flight.airline_flight', ',', 'flight.airline_code', ',', 'flight.flight_number', ',', 'flight.aircraft_code_sequence', ',', 'flight.meal_code', ',', 'flight.stops', ',', 'flight.connections', ',', 'flight.dual_carrier', ',', 'flight.time_elapsed', 'FROM', 'flight'], ['DISTINCT', 'flight.flight_id', ',', 'flight.flight_days', ',', 'flight.from_airport', ',', 'flight.to_airport', ',', 'flight.departure_time', ',', 'flight.arrival_time', ',', 'flight.airline_flight', ',', 'flight.airline_code', ',', 'flight.flight_number', ',', 'flight.aircraft_code_sequence', ',', 'flight.meal_code', ',', 'flight.stops', ',', 'flight.connections', ',', 'flight.dual_carrier', ',', 'flight.time_elapsed', 'FROM', 'flight'], ['DISTINCT', 'flight.flight_id', ',', 'flight', '.', 'flight_days', ',', 'flight', '.', 'from_airport', ',', 'flight', '.', 'to_airport', ',', 'flight', '.', 'departure_time', ',', 'flight', '.', 'arrival_time', ',', 'flight', '.', 'airline_flight', ',', 'flight', '.', 'airline_code', ',', 'flight', '.', 'flight_number', ',', 'flight', '.', 'aircraft_code_sequence', ',', 'flight', '.', 'meal_code', ',', 'flight', '.', 'stops', ',', 'flight', '.', 'connections', ',', 'flight', '.', 'dual_carrier', ',', 'flight', '.', 'time_elapsed', 'FROM', 'flight']], 
        #   'all_gold_queries': [(['select', 'documents.*'], [])], 
        #   'anonymized_gold_query': ['select', 'documents.*'], 
        #   'gold_query_to_use': ['select', 'documents.*']}
        # (((((((((((((((((((((((((((((((^^^^^^^^^^^^^^^^^^^^)))))))))))))))))))))))))))))))

        # create schema:
        # class Schema:
        #schema = Schema(self, table_schema, simple=False)
        from atis_data import read_database_schema
        # /Users/nicolasg-chausseau/editsql/database/data/cre_Doc_Template_Mgt/schema.sql
        database_schema, column_names_surface_form, column_names_embedder_input = read_database_schema("/Users/nicolasg-chausseau/editsql/database/data/sparc/tables.json")
        print("^^^^^^^^^^^^")
        print("vars(database_schema)")
        print(vars(database_schema))
        print("^^^^^^^^^^^^")

        # create Interaction
            # """ ATIS interaction class.

            # Attributes:
            #     utterances (list of Utterance): The utterances in the interaction.
            #     snippets (list of Snippet): The snippets that appear through the interaction.
            #     anon_tok_to_ent:
            #     identifier (str): Unique identifier for the interaction in the dataset.
            # """
        utterances = list_of_previous_utt + [example_utt_CLONE]
        sub_interaction = Interaction(
                utterances,
                schema,
                snippets,
                anon_tok_to_ent,
                identifier,
                params
            )
        example_inter = copy.deepcopy(interaction.interaction)
        utterances = example_inter.utterances 
        schema = example_inter.schema 
        snippets = example_inter.snippets 
        anon_tok_to_ent = example_inter.anon_tok_to_ent 
        identifier = example_inter.identifier 
        # ((((((((((((((((((((((((((((((~~~~~~~~~~~~~~~~~~~~))))))))))))))))))))))))))))))
        # vars(interaction.interaction)
        # { 'utterances': [<data_util.utterance.Utterance object at 0x7fedf8759588>, <data_util.utterance.Utterance object at 0x7fedf8759b70>, <data_util.utterance.Utterance object at 0x7fedf8759cc0>], 
        #   'schema': <data_util.interaction.Schema object at 0x7fedfb6fd1d0>, 
        #   'snippets': [], 
        #   'anon_tok_to_ent': {}, 
        #   'identifier': 'cre_Doc_Template_Mgt/0' }
        # (((((((((((((((((((((((((((((((^^^^^^^^^^^^^^^^^^^^)))))))))))))))))))))))))))))))

        # create interactionItem
        interaction = InteractionItem(
                interaction,
                max_input_length=float('inf'),
                max_output_length=float('inf'),
                nl_to_sql_dict={},
                maximum_length=float('inf')
                )

        # ^^^^^^^^^^^^^^^^^^^^
        # vars(interaction)
        # { 'interaction': <data_util.interaction.Interaction object at 0x7f263bdd1550>, 
        #   'processed_utterances': [], 
        #   'snippet_bank': [], 
        #   'identifier': 'cre_Doc_Template_Mgt/0', 
        #   'max_input_length': inf, 
        #   'max_output_length': inf, 
        #   'nl_to_sql_dict': <data_util.entities.NLtoSQLDict object at 0x7f269f2b4f28>, 
        #   'index': 0 }
        # ^^^^^^^^^^^^^^^^^^^^





def evaluate_interaction_sample_NIC(sample,
                                model,
                                max_generation_length,
                                name="",
                                gold_forcing=False,
                                metrics=None,
                                total_num=-1,
                                database_username="",
                                database_password="",
                                database_timeout=0,
                                use_predicted_queries=False,
                                write_results=False,
                                use_gpu=False,
                                compute_metrics=False):
    """ Evaluates a sample of interactions. """

    print(" INSIDE --> evaluate_interaction_sample_NIC")

    num_utterances = 0
    predictions = []

    model.eval()



    # # nic --> create a new interaction already initialized with most of the good stuff ... but fill it with the utterances from the (streamlit or cmdline) user:
    # #interaction = 
    for i, interaction_ in enumerate(sample):
        interaction = interaction_
        break
    print("^^^^^^^^^^^^^^^^^^^^")
    print("vars(interaction)")
    print(vars(interaction))
    print("^^^^^^^^^^^^^^^^^^^^")

    # ^^^^^^^^^^^^^^^^^^^^
    # vars(interaction)
    # { 'interaction': <data_util.interaction.Interaction object at 0x7f263bdd1550>, 
    #   'processed_utterances': [], 
    #   'snippet_bank': [], 
    #   'identifier': 'cre_Doc_Template_Mgt/0', 
    #   'max_input_length': inf, 
    #   'max_output_length': inf, 
    #   'nl_to_sql_dict': <data_util.entities.NLtoSQLDict object at 0x7f269f2b4f28>, 
    #   'index': 0 }
    # ^^^^^^^^^^^^^^^^^^^^

    print("((((((((((((((((((((((((((((((~~~~~~~~~~~~~~~~~~~~))))))))))))))))))))))))))))))")
    print("vars(interaction.interaction)")
    print(vars(interaction.interaction))
    print("vars(interaction.interaction.schema)")
    print(vars(interaction.interaction.schema))
    print("(((((((((((((((((((((((((((((((^^^^^^^^^^^^^^^^^^^^)))))))))))))))))))))))))))))))")

    # ((((((((((((((((((((((((((((((~~~~~~~~~~~~~~~~~~~~))))))))))))))))))))))))))))))
    # vars(interaction.interaction)
    # { 'utterances': [<data_util.utterance.Utterance object at 0x7fedf8759588>, <data_util.utterance.Utterance object at 0x7fedf8759b70>, <data_util.utterance.Utterance object at 0x7fedf8759cc0>], 
    #   'schema': <data_util.interaction.Schema object at 0x7fedfb6fd1d0>, 
    #   'snippets': [], 
    #   'anon_tok_to_ent': {}, 
    #   'identifier': 'cre_Doc_Template_Mgt/0' }
    # vars(interaction.interaction.schema)
    # {'table_schema': {
    #    'column_names': [
    #       [-1, '*'], [0, 'template type code'], [0, 'template type description'], [1, 'template id'], [1, 'version number'], [1, 'template type code'], [1, 'date effective from'], [1, 'date effective to'], [1, 'template details'], [2, 'document id'], [2, 'template id'], [2, 'document name'], [2, 'document description'], [2, 'other details'], [3, 'paragraph id'], [3, 'document id'], [3, 'paragraph text'], [3, 'other details']], 'column_names_original': [[-1, '*'], [0, 'Template_Type_Code'], [0, 'Template_Type_Description'], [1, 'Template_ID'], [1, 'Version_Number'], [1, 'Template_Type_Code'], [1, 'Date_Effective_From'], [1, 'Date_Effective_To'], [1, 'Template_Details'], [2, 'Document_ID'], [2, 'Template_ID'], [2, 'Document_Name'], [2, 'Document_Description'], [2, 'Other_Details'], [3, 'Paragraph_ID'], [3, 'Document_ID'], [3, 'Paragraph_Text'], [3, 'Other_Details']], 'column_types': ['text', 'text', 'text', 'number', 'number', 'text', 'time', 'time', 'text', 'number', 'number', 'text', 'text', 'text', 'number', 'number', 'text', 'text'], 'db_id': 'cre_Doc_Template_Mgt', 'foreign_keys': [[5, 1], [10, 3], [15, 9]], 'primary_keys': [1, 3, 9, 14], 'table_names': ['reference template types', 'templates', 'documents', 'paragraphs'], 'table_names_original': ['Ref_Template_Types', 'Templates', 'Documents', 'Paragraphs']}, 'column_names_surface_form': ['*', 'ref_template_types.template_type_code', 'ref_template_types.template_type_description', 'templates.template_id', 'templates.version_number', 'templates.template_type_code', 'templates.date_effective_from', 'templates.date_effective_to', 'templates.template_details', 'documents.document_id', 'documents.template_id', 'documents.document_name', 'documents.document_description', 'documents.other_details', 'paragraphs.paragraph_id', 'paragraphs.document_id', 'paragraphs.paragraph_text', 'paragraphs.other_details', 'ref_template_types.*', 'templates.*', 'documents.*', 'paragraphs.*'], 'column_names_surface_form_to_id': {'*': 0, 'ref_template_types.template_type_code': 1, 'ref_template_types.template_type_description': 2, 'templates.template_id': 3, 'templates.version_number': 4, 'templates.template_type_code': 5, 'templates.date_effective_from': 6, 'templates.date_effective_to': 7, 'templates.template_details': 8, 'documents.document_id': 9, 'documents.template_id': 10, 'documents.document_name': 11, 'documents.document_description': 12, 'documents.other_details': 13, 'paragraphs.paragraph_id': 14, 'paragraphs.document_id': 15, 'paragraphs.paragraph_text': 16, 'paragraphs.other_details': 17, 'ref_template_types.*': 18, 'templates.*': 19, 'documents.*': 20, 'paragraphs.*': 21}, 'column_names_embedder_input': ['*', 'reference template types . template type code', 'reference template types . template type description', 'templates . template id', 'templates . version number', 'templates . template type code', 'templates . date effective from', 'templates . date effective to', 'templates . template details', 'documents . document id', 'documents . template id', 'documents . document name', 'documents . document description', 'documents . other details', 'paragraphs . paragraph id', 'paragraphs . document id', 'paragraphs . paragraph text', 'paragraphs . other details', 'reference template types . *', 'templates . *', 'documents . *', 'paragraphs . *'], 'column_names_embedder_input_to_id': {'*': 0, 'reference template types . template type code': 1, 'reference template types . template type description': 2, 'templates . template id': 3, 'templates . version number': 4, 'templates . template type code': 5, 'templates . date effective from': 6, 'templates . date effective to': 7, 'templates . template details': 8, 'documents . document id': 9, 'documents . template id': 10, 'documents . document name': 11, 'documents . document description': 12, 'documents . other details': 13, 'paragraphs . paragraph id': 14, 'paragraphs . document id': 15, 'paragraphs . paragraph text': 16, 'paragraphs . other details': 17, 'reference template types . *': 18, 'templates . *': 19, 'documents . *': 20, 'paragraphs . *': 21}, 'num_col': 22}
    # (((((((((((((((((((((((((((((((^^^^^^^^^^^^^^^^^^^^)))))))))))))))))))))))))))))))

    print("((((((((((((((((((((((((((((((~~~~~~~~~~~~~~~~~~~~))))))))))))))))))))))))))))))")
    print("vars(interaction.interaction.utterances[0])")
    print(vars(interaction.interaction.utterances[0]))
    print("(((((((((((((((((((((((((((((((^^^^^^^^^^^^^^^^^^^^)))))))))))))))))))))))))))))))")

    # ((((((((((((((((((((((((((((((~~~~~~~~~~~~~~~~~~~~))))))))))))))))))))))))))))))
    # vars(interaction.interaction.utterances[0])
    # { 'original_input_seq': ['Show', 'information', 'for', 'all', 'documents', '.'], 
    #   'available_snippets': [], 
    #   'keep': True, 
    #   'input_seq_to_use': ['Show', 'information', 'for', 'all', 'documents', '.'], 
    #   'original_gold_query': ['select', 'documents.*'], 
    #   'gold_sql_results': [], 
    #   'contained_entities': [
    #       ['DISTINCT', 'flight.flight_id', ',', 'flight.flight_days', ',', 'flight.from_airport', ',', 'flight.to_airport', ',', 'flight.departure_time', ',', 'flight.arrival_time', ',', 'flight.airline_flight', ',', 'flight.airline_code', ',', 'flight.flight_number', ',', 'flight.aircraft_code_sequence', ',', 'flight.meal_code', ',', 'flight.stops', ',', 'flight.connections', ',', 'flight.dual_carrier', ',', 'flight.time_elapsed', 'FROM', 'flight'], ['DISTINCT', 'flight.flight_id', ',', 'flight.flight_days', ',', 'flight.from_airport', ',', 'flight.to_airport', ',', 'flight.departure_time', ',', 'flight.arrival_time', ',', 'flight.airline_flight', ',', 'flight.airline_code', ',', 'flight.flight_number', ',', 'flight.aircraft_code_sequence', ',', 'flight.meal_code', ',', 'flight.stops', ',', 'flight.connections', ',', 'flight.dual_carrier', ',', 'flight.time_elapsed', 'FROM', 'flight'], ['DISTINCT', 'flight.flight_id', ',', 'flight', '.', 'flight_days', ',', 'flight', '.', 'from_airport', ',', 'flight', '.', 'to_airport', ',', 'flight', '.', 'departure_time', ',', 'flight', '.', 'arrival_time', ',', 'flight', '.', 'airline_flight', ',', 'flight', '.', 'airline_code', ',', 'flight', '.', 'flight_number', ',', 'flight', '.', 'aircraft_code_sequence', ',', 'flight', '.', 'meal_code', ',', 'flight', '.', 'stops', ',', 'flight', '.', 'connections', ',', 'flight', '.', 'dual_carrier', ',', 'flight', '.', 'time_elapsed', 'FROM', 'flight']], 
    #   'all_gold_queries': [(['select', 'documents.*'], [])], 
    #   'anonymized_gold_query': ['select', 'documents.*'], 
    #   'gold_query_to_use': ['select', 'documents.*']}
    # (((((((((((((((((((((((((((((((^^^^^^^^^^^^^^^^^^^^)))))))))))))))))))))))))))))))

    # at this point ... it contains the utterances from the dev set!!!!!
    # self.interaction = copy.deepcopy(interaction)
    # self.interaction.utterances = self.interaction.utterances[:maximum_length]
    import copy
    example_utt = copy.deepcopy(interaction.interaction.utterances[1]) # so there's something in there that is still linking to the original question ... what is that property? how is it still linking to it?????
    
    interaction.interaction.utterances = []

    # assert: snippet_bank len is 0
    interaction.snippet_bank = [] # snippet_bank exists at the same level as processed_utterances: InteractionItem class
    #interaction.interaction.snippet_bank = [] 
    # assert len(self.processed_utterances) == 0
    interaction.processed_utterances = []
    # assert self.index == 0
    interaction.index = 0

    # self.identifier = self.interaction.identifier
    # self.nl_to_sql_dict = nl_to_sql_dict
    # self.index = 0

    # nic --> create a new utterance already initialized with the good stuff and modify it. 
    example_utt.original_gold_query = None# shortest_gold_and_results[0]
    example_utt.gold_sql_results = None#shortest_gold_and_results[1]
    example_utt.contained_entities = None#entities_in_input
    # Keep track of all gold queries and the resulting tables so that we can
    # give credit if it predicts a different correct sequence.
    example_utt.all_gold_queries = None # output_sequences
    example_utt.anonymized_gold_query = None#self.original_gold_query
    example_utt.gold_query_to_use = None
    example_utt.original_input_seq = None #tokenizers.nl_tokenize(example[params.input_key])
    example_utt.available_snippets = None
    example_utt.keep = False
    #snippet_bank
    import copy
    example_inter = copy.deepcopy(interaction.interaction)
    utterances = example_inter.utterances 
    schema = example_inter.schema 
    snippets = example_inter.snippets 
    anon_tok_to_ent = example_inter.anon_tok_to_ent 
    identifier = example_inter.identifier 



    #while True:
    list_of_all_utt = [
        # ['Show', 'the', 'template', 'id', 'for', 'all', 'documents', '.'],
        # ['Show', 'all', 'distinct', 'results', '.'],
        # ['How', 'many', 'are', 'there', '?']
        'Show the template id for all documents .',
        'Show all distinct results .',
        'How many are there ?'
    ]
    list_of_previous_utt = []
    counter = -1
    for utt in list_of_all_utt:
        counter += 1
        # 3
        # ///
        # ['Show', 'the', 'template', 'id', 'for', 'all', 'documents', '.']
        # ['Show', 'all', 'distinct', 'results', '.']
        # ['How', 'many', 'are', 'there', '?']
        # /------------
        # ------------
        # example_preds: too long
        # /------------
        # ------------
        # sequence: 
        # ['select', 'documents.template_id', '_EOS']
        # gold query: 
        # ['select', 'documents.template_id']
        # /------------
        # ------------
        # sequence: 
        # ['select', 'distinct', 'documents.template_id', '_EOS']
        # gold query: 
        # ['select', 'distinct', 'documents.template_id']
        # /------------
        # ------------
        # sequence: 
        # ['select', 'count', '(', 'distinct', 'documents.template_id', ')', '_EOS']
        # gold query: 
        # ['select', 'count', '(', 'distinct', 'documents.template_id', ')']
        # /------------
        # ------------

        # create utterance from old one:
        # def __init__(self,
        #              example,
        #              available_snippets,
        #              nl_to_sql_dict,
        #              params,
        #              anon_tok_to_ent={},
        #              anonymizer=None):
        #example = utt # or pass the whole list rather?
        
        available_snippets = example_inter.utterances[0].available_snippets
        nl_to_sql_dict = {}
        params = example_inter.utterances[0].params
        #params.input_key = example_inter.utterances[0].params.input_key #counter
        print("$$$$$$$$$$$")
        print("params.input_key")
        print(params.input_key)
        print("$$$$$$$$$$$")
        anon_tok_to_ent = example_inter.utterances[0].anon_tok_to_ent
        anonymizer = example_inter.utterances[0].anonymizer
        example = {params.input_key: utt} #list_of_previous_utt+[example_utt_CLONE]

        # example_utt_CLONE = Utterance(
        #         example,
        #         available_snippets,
        #         nl_to_sql_dict,
        #         params,
        #         anon_tok_to_ent,
        #         anonymizer)


        # create an Interaction object based on all necessary info:
        # class Interaction: 
        # def __init__(self,
        #              utterances,
        #              schema, # input_schema = interaction.get_schema(). # interaction.start_interaction() (interActionItem)
        #              snippets, 
        #              anon_tok_to_ent,
        #              identifier,
        #              params):

        # --create an utterance object for each current *and* previous question.
        interaction.interaction = Interaction(
                list_of_previous_utt+[example_utt_CLONE], #utterances,
                schema, # input_schema = interaction.get_schema(). # interaction.start_interaction() (interActionItem)
                snippets, 
                anon_tok_to_ent,
                identifier,
                params)


        # fix existing interactions or create a new one:
        # assert: snippet_bank len is 0
        interaction.snippet_bank = [] # snippet_bank exists at the same level as processed_utterances: InteractionItem class
        #interaction.interaction.snippet_bank = [] 
        # assert len(self.processed_utterances) == 0
        interaction.processed_utterances = []
        # assert self.index == 0
        interaction.index = 0

        # purpose is just to truncate and we dont want to truncate --> interaction.interaction.utterances = interaction.interaction.utterances[:]
        interaction.identifier = interaction.interaction.identifier # try None, if it's still doing the dev set
        print("**********")
        print("interaction.identifier")
        print(interaction.identifier)
        print("**********")
        interaction.nl_to_sql_dict = {}

        # class InteractionItem():
        #     def __init__(self,
        #                  interaction,
        #                  max_input_length=float('inf'),
        #                  max_output_length=float('inf'),
        #                  nl_to_sql_dict={},
        #                  maximum_length=float('inf')):
        #         if maximum_length != float('inf'):
        #             --self.interaction = copy.deepcopy(interaction)
        #             --TRUNCATE self.interaction.utterances = self.interaction.utterances[:maximum_length]
        #         else:
        #             self.interaction = interaction
        #         --self.processed_utterances = []
        #         --self.snippet_bank = []
        #         --self.identifier = self.interaction.identifier

        #         // self.max_input_length = max_input_length
        #         // self.max_output_length = max_output_length

        #         --self.nl_to_sql_dict = nl_to_sql_dict

        #         --self.index = 0



        #mystring = ... as string: inputted in stramlit UI --> ['Show', 'the', 'template', 'id', 'for', 'all', 'documents', '.']
        #interaction.utterances.append(['Show', 'the', 'template', 'id', 'for', 'all', 'documents', '.'])
        # from data_util import tokenizers
        # example_utt_CLONE = copy.deepcopy(example_utt)
        # example_utt_CLONE.original_input_seq = tokenizers.nl_tokenize(utt) # example[params.input_key].   # self.original_input_seq = tokenizers.nl_tokenize(example[params.input_key])
        # example_utt_CLONE.process_input_seq(params.anonymize,
        #                        anonymizer,
        #                        anon_tok_to_ent)

        # # Process the gold sequence
        # example_utt_CLONE.process_gold_seq(output_sequences,
        #                       nl_to_sql_dict,
        #                       self.available_snippets,
        #                       params.anonymize,
        #                       anonymizer,
        #                       anon_tok_to_ent)

        # def __init__(self,
        #              example,
        #              available_snippets,
        #              nl_to_sql_dict,
        #              params,
        #              anon_tok_to_ent={},
        #              anonymizer=None):
        #     # Get output and input sequences from the dictionary representation.
        #     output_sequences = example[OUTPUT_KEY]
        #     self.original_input_seq = tokenizers.nl_tokenize(example[params.input_key])
        #     self.available_snippets = available_snippets
        #     self.keep = False

        #     # pruned_output_sequences = []
        #     # for sequence in output_sequences:
        #     #     if len(sequence[0]) > 3:
        #     #         pruned_output_sequences.append(sequence)

        #     # output_sequences = pruned_output_sequences
        #     if len(output_sequences) > 0 and len(self.original_input_seq) > 0:
        #         # Only keep this example if there is at least one output sequence.
        #         self.keep = True
        #     if len(output_sequences) == 0 or len(self.original_input_seq) == 0:
        #         return

        #     # Process the input sequence
        #     self.process_input_seq(params.anonymize,
        #                            anonymizer,
        #                            anon_tok_to_ent)

        #     # Process the gold sequence
        #     self.process_gold_seq(output_sequences,
        #                           nl_to_sql_dict,
        #                           self.available_snippets,
        #                           params.anonymize,
        #                           anonymizer,
        #                           anon_tok_to_ent)


        # when it's ready with all the right properties and processed, pass it to predict on it:
        interaction.interaction.utterances.append(example_utt_CLONE) # make this work, needs to be the right type initialized correctly.

        print("------------NIC")
        print(i)
        print("///")
        print("utt: ")
        print(utt)
        print("---")
        #print(interaction.interaction.utterances) # self.original_input_seq
        for x in interaction.interaction.utterances:
            print(x.original_input_seq)
        print("/------------NIC")

        try:
            with torch.no_grad():
                if use_predicted_queries:
                    example_preds = model.predict_with_predicted_queries(   # the problem here is that in order to predict for a conversation / interaction, you need to pass the whole interaction in advance!!! that's nuts.
                        interaction,
                        max_generation_length)

                    print("------------NIC")
                    print("example_preds: too long")
                    # try:
                    #     print(example_preds)
                    # except Exception as e:
                    #     print(e)
                    print("/------------NIC")

                else:
                    example_preds = model.predict_with_gold_queries( 
                        interaction,
                        max_generation_length,
                        feed_gold_query=gold_forcing)
                torch.cuda.empty_cache()
        except RuntimeError as exception:
            print("Failed on interaction: " + str(interaction.identifier))
            print(exception)
            print("\n\n")
            exit()

        predictions.extend(example_preds)

        assert len(example_preds) == len(
            interaction.interaction.utterances) or not example_preds
        for j, pred in enumerate(example_preds):
            num_utterances += 1

            sequence, loss, token_accuracy, _, decoder_results = pred

            print("------------")
            print("sequence: ")
            print(sequence)
            print("original_gold_query: ")
            print(interaction.interaction.utterances[j].original_gold_query)
            print("/------------")

    list_of_previous_utt.append(example_utt_CLONE)












def evaluate_interaction_sample(sample,
                                model,
                                max_generation_length,
                                name="",
                                gold_forcing=False,
                                metrics=None,
                                total_num=-1,
                                database_username="",
                                database_password="",
                                database_timeout=0,
                                use_predicted_queries=False,
                                write_results=False,
                                use_gpu=False,
                                compute_metrics=False):
    """ Evaluates a sample of interactions. """

    print("----------")
    print("--> INSIDE evaluate_interaction_sample")
    print("name: ")
    print(name)
    print("----------")

    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with file " + str(name + "_predictions.json"))
    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    num_utterances = 0
    ignore_with_gpu = [line.strip() for line in open(
        "data/cpu_full_interactions.txt").readlines()]
    predictions = []

    use_gpu = not ("--no_gpus" in sys.argv or "--no_gpus=1" in sys.argv)

    model.eval()

    for i, interaction in enumerate(sample):
        # if use_gpu and interaction.identifier in ignore_with_gpu:
        #     continue
        # elif not use_gpu and interaction.identifier not in ignore_with_gpu:
        #     continue

        print("------------")
        print(i)
        print("///")
        #print(interaction.interaction.utterances) # self.original_input_seq
        for x in interaction.interaction.utterances:
            print(x.original_input_seq)
        print("/------------")

        try:
            with torch.no_grad():
                if use_predicted_queries:
                    example_preds = model.predict_with_predicted_queries(   # the problem here is that in order to predict for a conversation / interaction, you need to pass the whole interaction in advance!!! that's nuts.
                        interaction,
                        max_generation_length)

                    print("------------")
                    print("example_preds: too long")
                    # try:
                    #     print(example_preds)
                    # except Exception as e:
                    #     print(e)
                    print("/------------")

                else:
                    example_preds = model.predict_with_gold_queries( 
                        interaction,
                        max_generation_length,
                        feed_gold_query=gold_forcing)
                torch.cuda.empty_cache()
        except RuntimeError as exception:
            print("Failed on interaction: " + str(interaction.identifier))
            print(exception)
            print("\n\n")
            exit()

        predictions.extend(example_preds)

        assert len(example_preds) == len(
            interaction.interaction.utterances) or not example_preds
        for j, pred in enumerate(example_preds):
            num_utterances += 1

            sequence, loss, token_accuracy, _, decoder_results = pred

            print("------------")
            print("sequence: ")
            print(sequence)
            print("original_gold_query: ")
            print(interaction.interaction.utterances[j].original_gold_query)
            print("/------------")


            if use_predicted_queries:
                item = interaction.processed_utterances[j]
                original_utt = interaction.interaction.utterances[item.index]

                gold_query = original_utt.gold_query_to_use
                original_gold_query = original_utt.original_gold_query

                gold_table = original_utt.gold_sql_results
                gold_queries = [q[0] for q in original_utt.all_gold_queries]
                gold_tables = [q[1] for q in original_utt.all_gold_queries]
                index = item.index
            else:
                item = interaction.gold_utterances()[j]

                gold_query = item.gold_query()
                original_gold_query = item.original_gold_query()

                gold_table = item.gold_table()
                gold_queries = item.original_gold_queries()
                gold_tables = item.gold_tables()
                index = item.utterance_index
            if loss:
                loss = loss / len(gold_query)

            flat_sequence = item.flatten_sequence(sequence)

            if write_results:
                write_prediction(
                    predictions_file,
                    identifier=interaction.identifier,
                    input_seq=item.input_sequence(),
                    probability=decoder_results.probability,
                    prediction=sequence,
                    flat_prediction=flat_sequence,
                    gold_query=gold_query,
                    flat_gold_queries=gold_queries,
                    gold_tables=gold_tables,
                    index_in_interaction=index,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    compute_metrics=compute_metrics)

            update_sums(metrics,
                        metrics_sums,
                        sequence,
                        flat_sequence,
                        gold_query,
                        original_gold_query,
                        gold_forcing,
                        loss,
                        token_accuracy,
                        database_username=database_username,
                        database_password=database_password,
                        database_timeout=database_timeout,
                        gold_table=gold_table)

        progbar.update(i)

    progbar.finish()

    if total_num < 0:
        total_num = num_utterances

    predictions_file.close()
    return construct_averages(metrics_sums, total_num), predictions


def evaluate_using_predicted_queries(sample,
                                     model,
                                     name="",
                                     gold_forcing=False,
                                     metrics=None,
                                     total_num=-1,
                                     database_username="",
                                     database_password="",
                                     database_timeout=0,
                                     snippet_keep_age=1):

    print("--> INSIDE evaluate_using_predicted_queries")

    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with file " + str(name + "_predictions.json"))
    assert not gold_forcing
    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    num_utterances = 0
    predictions = []
    for i, item in enumerate(sample):

        print("------------")
        print(i)
        print("///")
        #print(item)
        print("/------------")

        int_predictions = []
        item.start_interaction()
        while not item.done():

            print("-----------")
            print("snippet_keep_age: ")
            print(snippet_keep_age)
            print("-----------")

            # utterance = item.next_utterance(snippet_keep_age) # TypeError: next_utterance() takes 1 positional argument but 2 were given
            utterance = item.next_utterance()
    
            print("------------")
            print(utterance)
            print("/------------")

            predicted_sequence, loss, _, probability = model.eval_step(
                utterance)

            print("----------")
            print("predicted_sequence: ")
            print(predicted_sequence)
            print("----------")

            int_predictions.append((utterance, predicted_sequence))

            flat_sequence = utterance.flatten_sequence(predicted_sequence)

            if sql_util.executable(
                    flat_sequence,
                    username=database_username,
                    password=database_password,
                    timeout=database_timeout) and probability >= 0.24:
                utterance.set_pred_query(
                    item.remove_snippets(predicted_sequence))
                item.add_utterance(utterance,
                                   item.remove_snippets(predicted_sequence),
                                   previous_snippets=utterance.snippets())
            else:
                # Add the /previous/ predicted query, guaranteed to be syntactically
                # correct
                seq = []
                utterance.set_pred_query(seq)
                item.add_utterance(
                    utterance, seq, previous_snippets=utterance.snippets())

            original_utt = item.interaction.utterances[utterance.index]
            write_prediction(
                predictions_file,
                identifier=item.interaction.identifier,
                input_seq=utterance.input_sequence(),
                probability=probability,
                prediction=predicted_sequence,
                flat_prediction=flat_sequence,
                gold_query=original_utt.gold_query_to_use,
                flat_gold_queries=[
                    q[0] for q in original_utt.all_gold_queries],
                gold_tables=[
                    q[1] for q in original_utt.all_gold_queries],
                index_in_interaction=utterance.index,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout)

            update_sums(metrics,
                        metrics_sums,
                        predicted_sequence,
                        flat_sequence,
                        original_utt.gold_query_to_use,
                        original_utt.original_gold_query,
                        gold_forcing,
                        loss,
                        token_accuracy=0,
                        database_username=database_username,
                        database_password=database_password,
                        database_timeout=database_timeout,
                        gold_table=original_utt.gold_sql_results)

        predictions.append(int_predictions)
        progbar.update(i)

    progbar.finish()

    if total_num < 0:
        total_num = num_utterances
    predictions_file.close()

    return construct_averages(metrics_sums, total_num), predictions
