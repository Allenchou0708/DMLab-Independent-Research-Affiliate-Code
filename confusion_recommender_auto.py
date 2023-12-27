import pickle # read for binary
import torch
import argparse  # 用於撰寫 --參數
import random
import mmap
from tqdm import tqdm
from statistics import mean
from collections import defaultdict ## defaultdict["key"] += 1 => 計數用的Key-Value Pair
from os import mkdir
import pandas as pd
import numpy as np
import os
import glob
import random
from datetime import datetime
import constants.consts as consts
from model.kprn import KPRN
from model.predictor import predict
from data.format import format_paths
from model.train import train
from data.path_extraction import find_paths_user_to_songs
from eval import hit_at_k, ndcg_at_k
import copy
from try_code.self_print import show, show_type

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=False,
                        action='store_true', ## 如果有就設定成True
                        help='whether to train the model')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--find_paths',
                        default=False,
                        action='store_true',
                        help='whether to find paths (otherwise load from disk)') ###
    parser.add_argument('--subnetwork',
                        default='dense',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to load data from') ###
    parser.add_argument('--model',
                        type=str,
                        default='model.pt',
                        help='name to save or load model from') ###
    parser.add_argument('--load_checkpoint',
                        default=False,
                        action='store_true',
                        help='whether to load the current model state before training ') ###
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='interactions.txt',
                        help='file name to store/load train/test paths') ###

    parser.add_argument('--test_kg_path_file',
                        type=str,
                        default='interactions.txt',
                        help='file name to store/load train/test paths') ###

    parser.add_argument('--user_limit',
                        type=int,
                        default=10,
                        help='max number of users to find paths for')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=5,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('--not_in_memory',
                        default=False,
                        action='store_true',
                        help='denotes that the path data does not fit in memory')
    parser.add_argument('--lr',
                        type=float,
                        default=.002,
                        help='learning rate')
    parser.add_argument('--l2_reg',
                        type=float,
                        default=.0001,
                        help='l2 regularization coefficent')
    parser.add_argument('--gamma',
                        type=float,
                        default=1,
                        help='gamma for weighted pooling')
    parser.add_argument('--no_rel',
                        default=False,
                        action='store_true',
                        help='Run the model without relation if True')
    parser.add_argument('--np_baseline',
                        default=False,
                        action='store_true',
                        help='Run the model with the number of path baseline if True') ### with dense
    parser.add_argument('--samples',
                        type=int,
                        default=-1,
                        help='number of paths to sample for each interaction (-1 means include all paths)')
    parser.add_argument('--len3_branch',
                        type=int,
                        default=10,
                        help='number of paths to sample for each interaction (-1 means include all paths)')
    parser.add_argument('--len5_branch',
                        type=int,
                        default=10,
                        help='number of paths to sample for each interaction (-1 means include all paths)')

    return parser.parse_args() ### return.參數名 (ex: --samples) => 可以列出該參數的值


def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def sample_paths(paths, samples):
    index_list = list(range(len(paths)))
    random.shuffle(index_list)
    indices = index_list[:samples]
    return [paths[i] for i in indices]

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

# neg_samples = 4, len_3 = 50 , len_5 = 6
def load_data(user_word,word_word,word_user,user_gender,
              user_gender_split, gender_user_split, neg_samples, e_to_ix, t_to_ix,
              r_to_ix, kg_path_file, len_3_branch, len_5_branch, limit=10, version="train", samples=-1):
    '''
    Constructs paths for train/test data,

    For training, we write each formatted interaction to a file as we find them
    For testing, for each combo of a pos paths and 100 neg paths we store these in a single line in the file
    '''
  #  print("in loading data len : ",len(list(word_word.items())))

    first_len = 3
    last_len = 5

    currentDateAndTime = datetime.now()
    path_dir = 'data/' + consts.PATH_DATA_DIR
    create_directory(path_dir)
    path_file = open(path_dir + version + "_confusion_len"+str(first_len)+"_"+str(len_3_branch)+"_len"+str(last_len)+"_"+str(len_5_branch)+"_"+str(currentDateAndTime.day)+"_"+str(currentDateAndTime.hour)+"_"+str(currentDateAndTime.minute), 'w')

    #trackers for statistics
    pos_paths_not_found = 0
    total_pos_interactions = 0
    num_neg_interactions = 0
    avg_num_pos_paths, avg_num_neg_paths = 0, 0

    all_genders  = set()
    for gender_id in gender_user_split.keys():
        all_genders.add(gender_id)
    
    s_genders = sorted(list(all_genders))
    print("genders : ",s_genders)

    print("len user_gender : ",len(list(user_gender_split.items())))

    ## !! path for only limit

    
    if limit == -1 : 
        training_material = list(user_gender_split.items())
    else : 
        training_material  = list(user_gender_split.items())[:limit]

    runtime = -1
    print_time = 0
    for user,gender in tqdm(list(training_material)):
        runtime+=1
        gender = gender[0]
#        print(user," ",gender)
        cur_index = 0

        complete_paths, complete_neg_paths = None, None
        """
        total_pos_interactions += len(pos_songs) # user listen to these songs (user,[list of songs])
        cur_index = 0 #current index in negative list for adding negative interactions
        """

        interactions = [] #just used with finding test paths

        if version == "train":
            song_to_paths = find_paths_user_to_songs(user, user_word, word_word, word_user, user_gender, first_len, len_3_branch,version)
            song_to_paths_len5 = find_paths_user_to_songs(user, user_word, word_word, word_user, user_gender, last_len, len_5_branch,version)
        else: #for testing we use entire song_user and user_song dictionaries
            song_to_paths = find_paths_user_to_songs(user, user_word, word_word,word_user, user_gender, first_len, len_3_branch,version)
            song_to_paths_len5 = find_paths_user_to_songs(user, user_word, word_word, word_user, user_gender, last_len, len_5_branch,version)
        
        for song in song_to_paths_len5.keys():
            song_to_paths[song].extend(song_to_paths_len5[song])

   #     for k,v in list(song_to_paths.items())[:10]:
    #        print(k," ",v)

        #select negative paths
        neg_genders = all_genders.difference(set([gender]))
        pos_paths = song_to_paths[gender] # user 到這一首歌的path
        neg_paths = song_to_paths[list(neg_genders)[0]] ## 找的path,但和user沒有關係
    
        """
        if runtime < 3 :
            print("gender : ",gender)
            print("other gender : ",list(neg_genders)[0])
            print("pos paths : ",pos_paths[0])
            print("pos paths : ",pos_paths[1])
            print("neg paths : ",neg_paths[0])
            print("neg paths : ",neg_paths[1])
        """
  #      print("gender = ",gender)

#        print("     pos paths error : ")
        true_pos_paths = copy.copy(pos_paths)
        for path in pos_paths : 
            if path[-1][0] != gender :
         #       print(path)
                true_pos_paths.remove(path)
      
        true_neg_paths = copy.copy(neg_paths)  
#        print("     neg paths error : ")
        for path in neg_paths : 
            if path[-1][0] != list(neg_genders)[0] : 
         #       print(path)
                true_neg_paths.remove(path)

        pos_paths = true_pos_paths
        top_neg_songs = true_neg_paths
        random.shuffle(top_neg_songs)

#        print("done")

#        print("     pos paths error : ")
  #      print("     pos lens : ",len(pos_paths))
   #     for path in pos_paths : 
    #        if path[-1][0] != gender : 
     #           print(path)
        
 #       print("     neg paths error : ")
#        print("     neg lens : ",len(top_neg_songs))
  #      for path in top_neg_songs : 
   #         if path[-1][0] != list(neg_genders)[0] : 
    #            print(path)

        choosed_number = len(pos_paths) if len(pos_paths) < len(top_neg_songs) else len(top_neg_songs)

#        print("pos path : ",pos_paths)
 #       print("formated pos path : ",format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix))

        if version == "train" :
            
            if len(pos_paths) > 0:
                pos_paths = pos_paths[:choosed_number]
            
            
            if  len(top_neg_songs) > 0  :
                selected_neg_songs = top_neg_songs[:choosed_number]

            

            #add paths for positive interaction
            if len(pos_paths) > 0: 
                interaction = (format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1) ## format_path return [(new_path_format,len),(new_path_format,len)..]         
                path_file.write(repr(interaction) + "\n") # repr : 轉換為字串
            
          #  print("pos : ")
          #  print(interaction)
                
            if len(selected_neg_songs) > 0 : 
                interaction = (format_paths(selected_neg_songs, e_to_ix, t_to_ix, r_to_ix), 0) ## format_path return [(new_path_format,len),(new_path_format,len)..]
                path_file.write(repr(interaction) + "\n") # repr : 轉換為字串

          #  print("neg : ")
          #  print(interaction)
        else : 
            ## eval code:wq

            selected_interaction = []
            selected_neg_songs = []
            
            if len(pos_paths) > 0:
                pos_paths = pos_paths[:choosed_number]
            
            
            if  len(top_neg_songs) > 0  :
                selected_neg_songs = top_neg_songs[:choosed_number]


            #add paths for positive interaction
            if len(pos_paths) > 0: 
                interaction = (format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1) ## format_path return [(new_path_format,len),(new_path_format,len)..]         
                selected_interaction.append(interaction)
                
            if len(selected_neg_songs) > 0 : 
                interaction = (format_paths(selected_neg_songs, e_to_ix, t_to_ix, r_to_ix), 0) ## format_path return [(new_path_format,len),(new_path_format,len)..]
                selected_interaction.append(interaction)
            

            if len(pos_paths) > 0 and len(selected_neg_songs) > 0 : 
#                print(selected_interaction)
                path_file.write(repr( selected_interaction) + "\n")
    """
    avg_num_neg_paths = avg_num_neg_paths / num_neg_interactions
    avg_num_pos_paths = avg_num_pos_paths / (total_pos_interactions - pos_paths_not_found)
    """

    print("number of pos paths attempted to find:", total_pos_interactions)
    print("number of pos paths not found:", pos_paths_not_found)
    
    """
    print("avg num paths per positive interaction:", avg_num_pos_paths)
    print("avg num paths per negative interaction:", avg_num_neg_paths)
    """

    path_file.close()
    print("kg file saved at : ", path_dir + version + "_confusion_len3_"+str(len_3_branch)+"_len5_"+str(len_5_branch)+"_"+str(currentDateAndTime.day)+"_"+str(currentDateAndTime.hour)+"_"+str(currentDateAndTime.minute) )

    return


def load_string_to_ix_dicts(network_type):
    '''
    Loads the dictionaries mapping entity, relation, and type to id
    '''
    data_path = 'data/' + consts.SONG_IX_MAPPING_DIR + network_type

    with open(data_path + '_type_to_ix.dict', 'rb') as handle:
        type_to_ix = pickle.load(handle)
    with open(data_path + '_relation_to_ix.dict', 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(data_path + '_entity_to_ix.dict', 'rb') as handle:
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix


def load_rel_ix_dicts(network_type):
    '''
    Loads the relation dictionaries
    '''
    data_path = 'data/' + consts.SONG_IX_DATA_DIR + network_type

    with open(data_path + '_ix_user_keywords.pkl', 'rb') as handle:
        user_word = pickle.load(handle)
    with open(data_path + '_ix_word_word.dict', 'rb') as handle:
        word_word = pickle.load(handle)
    with open(data_path + '_ix_word_user.dict', 'rb') as handle:
        word_user = pickle.load(handle)
    with open(data_path + '_ix_user_gender_dict', 'rb') as handle:
        user_gender = pickle.load(handle)

    return user_word, word_word, word_user, user_gender


def main():
    '''
    Main function for kprn model testing and training
    '''
    print("Main Loaded")
    random.seed(1)   ###  Random Seed Here
    args = parse_args()

    if args.eval == True and not args.find_paths: 
        
        model_name = args.model
        model_path = "model/" + model_name
    else :
        model_path = "model/" + args.model

    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts(args.subnetwork) # 編碼表
    user_word,word_word,word_user,user_gender = load_rel_ix_dicts(args.subnetwork)
    
#    print("in recommander len : ",len(list(word_word.items())))


    model = KPRN(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                 len(e_to_ix), len(t_to_ix), len(r_to_ix), consts.TAG_SIZE, args.no_rel)

#    print("e_to_ix len : ",len(e_to_ix))
 #   print("t_to_ix len : ",len(t_to_ix))
  #  print("r_to_ix len : ",len(r_to_ix))


    data_ix_path = 'data/' + consts.SONG_IX_DATA_DIR + args.subnetwork
    

    if args.train:
        print("Training Starting")
        
        if args.find_paths: ## 如果參數指定要重建圖
            print("Finding paths")

            with open(data_ix_path + '_train_ix_user_gender_dict', 'rb') as handle:
                user_gender_train = pickle.load(handle)
            with open(data_ix_path + '_train_ix_gender_user_dict', 'rb') as handle:
                gender_user_train = pickle.load(handle)
            

            """            
            print("user_gender_train :")
            #print(type(user_gender_train))
            print(len(user_gender_train))
            print(list(user_gender_train.items())[:10])
            """

            load_data(user_word,word_word,word_user,user_gender,
                      user_gender_train, gender_user_train, consts.NEG_SAMPLES_TRAIN,
                      e_to_ix, t_to_ix, r_to_ix, args.kg_path_file, args.len3_branch,
                      args.len5_branch, limit=args.user_limit, version="train", samples=args.samples)

            exit()

        if not args.find_paths:
            kg_path_file = args.kg_path_file

        with open(data_ix_path + '_train_ix_gender_user_dict', 'rb') as handle:
            gender_user_train = pickle.load(handle)
        
        genders = list(gender_user_train.keys())
        print("total has ",len(genders)," genders")
        male = genders[0] if genders[0]<genders[1] else genders[1]
        female = genders[0] if male == genders[1] else genders[1]

        model = train(model, kg_path_file, args.batch_size, args.epochs, model_path,
                      args.load_checkpoint, args.not_in_memory, args.lr, args.l2_reg, args.gamma, args.no_rel,args.test_kg_path_file,male,female)

    if args.eval:
        print("Evaluation Starting")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is : ", device)

        if not args.find_paths:
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            model = model.to(device)

        if args.find_paths:
            print("Finding Paths")

            with open(data_ix_path + '_test_ix_user_gender_dict', 'rb') as handle:
                user_gender_test = pickle.load(handle)
            with open(data_ix_path + '_test_ix_gender_user_dict', 'rb') as handle:
                gender_user_test = pickle.load(handle)

            load_data(user_word,word_word,word_user,user_gender,
                      user_gender_test,gender_user_test, consts.NEG_SAMPLES_TRAIN,
                      e_to_ix, t_to_ix, r_to_ix, args.kg_path_file, args.len3_branch,
                      args.len5_branch, limit=args.user_limit, version="test", samples=args.samples)
            exit()

        if not args.find_paths:
            kg_path_file = args.kg_path_file


        #predict scores using model for each combination of one pos and 100 neg interactions
        hit_at_k_scores = defaultdict(list)
        ndcg_at_k_scores = defaultdict(list)
        if args.np_baseline:
            num_paths_baseline_hit_at_k = defaultdict(list)
            num_paths_baseline_ndcg_at_k = defaultdict(list)
        max_k = 15

        runtime=0
        file_path = 'data/path_data/' + kg_path_file

        total_test_number = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        my_test_interaction = []
        num_file = sum([1 for i in open(file_path,"r")])
        with open(file_path, 'r') as file :
            for line in tqdm(file,total=num_file) :
           # for line in tqdm(file,total=len(file.readlines())): ## total 可以顯示完整進度條
                my_test_interaction = eval(line.rstrip("\n"))
                prediction_scores = predict(model, my_test_interaction, args.batch_size, device, args.no_rel, args.gamma,file_path,args.not_in_memory) ## file_path / --nomemeory
                #print("in recommender : prediction score = ",prediction_scores," shape = ",len(prediction_scores))
                target_scores = [x[1] for x in my_test_interaction]

                #merge prediction scores and target scores into tuples, and rank
                merged = list(zip(prediction_scores, target_scores))
                for ps,ts in merged :
                    total_test_number += 1
                    
                    if ps >= 0.5 :
                        if ts == 1 :
                            TP += 1
                        else :
                            FP += 1
                    else :
                        if ts == 1 :
                            FN += 1
                        else :
                            TN += 1

                s_merged = sorted(merged, key=lambda x: x[0], reverse=True) #依照prediction由高到低

                """
                print("s_merged : ")
                if runtime < 10 :
                    print(s_merged)
                runtime+=1
                """

                for k in range(1,max_k+1):
                    hit_at_k_scores[k].append(hit_at_k(s_merged, k))
                    ndcg_at_k_scores[k].append(ndcg_at_k(s_merged, k))

                #Baseline of ranking based on number of paths
                if args.np_baseline:
                    random.shuffle(test_interactions)
                    s_inters = sorted(test_interactions, key=lambda x: len(x[0]), reverse=True) #取 prediction_scores 由大到小
                    for k in range(1,max_k+1):
                        num_paths_baseline_hit_at_k[k].append(hit_at_k(s_inters, k))
                        num_paths_baseline_ndcg_at_k[k].append(ndcg_at_k(s_inters, k))

        scores = []
        
        print("\ndata number : ", total_test_number )
        print()
        print("TP : ",TP," FP : ",FP," TN : ",TN," FN : ",FN)
        accuracy = (TP + TN) / total_test_number 
        print("accuracy = ", (TP + TN) / total_test_number )
        
        try : 
            precision = TP / (TP + FP)
            print("precision = ", precision)
        except :
            precision = -1
        
        try : 
            recall = TP / (TP + FN)
            print("recall = ",recall )
        except :
            recall = -1
       
        try:
            F_1 =  2 / (1/precision) + (1/recall)
            print("F1 = ", 2 / (1/precision) + (1/recall))
        except :
            F_1 = -1

        # saving scores
        scores_cols = ['model', 'test_file', 'k', 'hit', 'ndcg']
        scores_df = pd.DataFrame(scores, columns = scores_cols)
        scores_path = 'model_scores.csv'
        try:
            model_scores = pd.read_csv(scores_path)
        except FileNotFoundError:
            model_scores = pd.DataFrame(columns=scores_cols)
        model_scores=model_scores.append(scores_df, ignore_index = True, sort=False)
        model_scores.to_csv(scores_path,index=False)

        my_score = []
        my_score.append([model_name, kg_path_file, accuracy, precision,recall,F_1])
        scores_col = ['model', 'test_file', 'accuracy','precision','recall','F1']
        scores_rol = pd.DataFrame(my_score, columns = scores_col)
        score_path = 'model_scores_for_accuracy.csv'
        try : 
            old_model_scores = pd.read_csv(score_path)
        except FileNotFoundError:
            old_model_scores = pd.DataFrame(columns=scores_col)
        old_model_scores = old_model_scores.append(scores_rol, ignore_index = True, sort=False)
        old_model_scores.to_csv(score_path,index=False)


if __name__ == "__main__":
    main()
