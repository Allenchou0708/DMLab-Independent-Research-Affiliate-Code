import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import linecache

import constants.consts as consts
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean
import csv
import pandas as pd

class TrainInteractionData(Dataset):
    '''
    Dataset that can either store all interaction data in memory or load it line
    by line when needed
    '''
    def __init__(self, train_path_file, in_memory=True):
        self.in_memory = in_memory
        self.file = 'data/path_data/' + train_path_file
        self.num_interactions = 0
        self.interactions = []
        if in_memory:
            with open(self.file, "r") as f:
                for line in f:
                    self.interactions.append(eval(line.rstrip("\n"))) ## 轉為運算式
            self.num_interactions = len(self.interactions)
        else:
            with open(self.file, "r") as f:
                for line in f:
                    self.num_interactions += 1

    def __getitem__(self, idx):
        #load the specific interaction either from memory or from file line
        if self.in_memory:
            return self.interactions[idx]
        else:
            line = linecache.getline(self.file, idx+1) ## 會將以讀完的放到memory 讓每次讀取的時間降低 (index from 1 )
            return eval(line.rstrip("\n"))


    def __len__(self):
        return self.num_interactions


class TestInteractionData(Dataset):
    def __init__(self, formatted_data,file_path,in_memory=True):
        self.file = file_path
        self.num_interactions = 0
        self.in_memory = in_memory
    
        if in_memory : 
            self.data = formatted_data
        else : 
            with open(self.file, "r") as f:
                for line in f:
                    self.num_interactions += 1

    def __getitem__(self, index):

        if self.in_memory :
            return self.data[index]
        else : 
            line = linecache.getline(self.file, index+1) ## 會將以讀完的放到memory 讓每次讀取的時間降低 (index from 1 )
            return eval(line.rstrip("\n"))

    def __len__(self):
        if self.in_memory :
            return len(self.data)
        else : 
            return self.num_interactions



def convert_to_etr(e_to_ix, t_to_ix, r_to_ix, path, length):
    '''
    Converts a path of ids back to the original input format
    -not used for anything right now but could be useful for visualization
    '''
    ix_to_t = {v: k for k, v in t_to_ix.items()}
    ix_to_r = {v: k for k, v in r_to_ix.items()}
    ix_to_e = {v: k for k, v in e_to_ix.items()}
    new_path = []
    for i,step in enumerate(path):
        if i == length:
            break
        new_path.append([ix_to_e[step[0].item()], ix_to_t[step[1].item()], ix_to_r[step[2].item()]])
    return new_path


def my_collate(batch): ## 對每個 batch 先做出前處理
    '''
    Custom dataloader collate function since we have tuples of lists of paths
    '''
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target) ## tensor 內皆為long 型態
    return [data, target]


def sort_batch(batch, indexes, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    indexes_tensor = indexes[perm_idx]
    return seq_tensor, indexes_tensor, seq_lengths



def predict(model, formatted_data, batch_size, device, no_rel, gamma, testing_file, not_in_memory):
    '''
    -outputs predicted scores for the input test data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    -Since we are evaluating we ignore the tag here
    '''

    prediction_scores = []

    #print("not not_in_memory : ",not not_in_memory)
    interaction_data = TestInteractionData(formatted_data,file_path = testing_file,in_memory = not not_in_memory) ## 某行測資
    #shuffle false since we want data to remain in order for comparison
    test_loader = DataLoader(dataset=interaction_data, collate_fn = my_collate, batch_size=batch_size, shuffle=False)
   
    #print("test_loader len : ",len(test_loader))

    with torch.no_grad():
        for (interaction_batch, _) in test_loader:
            #construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
            paths = []
            lengths = []
            inter_ids = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)


            #sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)
            
            #print(" inter_ids : " , inter_ids)
            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel)

            #Get weighted pooling of scores over interaction id groups
            start = True
            for i in range(len(interaction_batch)):
                #get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)

                #weighted pooled scores for this interaction
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                if start:
                    #unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

            prediction_scores.extend(F.softmax(pooled_scores, dim=1))

#            print("prediction_scores len : ",len(prediction_scores)," it is : ",prediction_scores[0])

    #just want positive scores currently
    pos_scores = []
    for tensor in prediction_scores:
        pos_scores.append(tensor.tolist()[1])
    return pos_scores


def train(model, train_path_file, batch_size, epochs, model_path, load_checkpoint,
         not_in_memory, lr, l2_reg, gamma, no_rel,test_path_file,male_id,female_id):
    '''
    -trains and outputs a model using the input data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("Device is", device)
    model = model.to(device)
    
    model_path = "model/" +  "lr_"+str(lr)+"_l2_"+str(l2_reg)+"_e_"+str(epochs)+"_bs_"+str(batch_size)+"_g_"+str(gamma)+".pt"
    model_name = "lr_"+str(lr)+"_l2_"+str(l2_reg)+"_e_"+str(epochs)+"_bs_"+str(batch_size)+"_g_"+str(gamma)
    loss_function = nn.NLLLoss() ## Negative Log Loss

    # l2 regularization is tuned from {10−5 , 10−4 , 10−3 , 10−2 }, I think this is weight decay
    # Learning rate is found from {0.001, 0.002, 0.01, 0.02} with grid search
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    if load_checkpoint:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #DataLoader used for batches
    interaction_data = TrainInteractionData(train_path_file, in_memory=not not_in_memory)
    train_loader = DataLoader(dataset=interaction_data, collate_fn = my_collate, batch_size=batch_size, shuffle=True)

    x_axis = []
    y_axis = []

    epoch_train_score = []
    epoch_test_score = []

    best_test_accuracy = 0

    
    for epoch in range(epochs):
        
        print("Epoch is:", epoch+1)
        losses = []
        for interaction_batch, targets in tqdm(train_loader): #have tqdm here when not on colab
            #construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
            paths = []
            lengths = []
            inter_ids = [] ## 可行路徑的集合
            for inter_id, interaction_paths in enumerate(interaction_batch): ## interaction_batch = [paths]
                for path, length in interaction_paths: ## paths = [(Fulled 2d path ,original_len)]
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))]) ## extend : 將list2 的元素抽出, 放到list1內

            

            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)
            

            #sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            #Pytorch accumulates gradients, so we need to clear before each instance
            model.zero_grad()

            #Run the forward pass.
            tag_scores = model(s_path_batch.to(device), s_lengths.to('cpu'), no_rel)

            #Get weighted pooling of scores over interaction id groups
            start = True
            for i in range(len(interaction_batch)):
                #get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1) #選出同樣 input 的 index , 並成為1維

                #weighted pooled scores for this interaction
                #if i ==0 : 
                 #   print("tag_scores : ",tag_scores[inter_idxs])
                #elif i == 10 : 
                #    exit()
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma) ## selected_size * 2

                if start:
                    #unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)
               #     print("pooled_scores : ",pooled_scores)
            
            #print("pooled_scores : ",pooled_scores)
            prediction_scores = F.log_softmax(pooled_scores, dim=1)
            #print("prediction_scores : ",prediction_scores)

            #Compute the loss, gradients, and update the parameters by calling .step()
            loss = loss_function(prediction_scores.to(device), targets.to(device))
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

        
        print("male : ",male_id," female : ",female_id)        
        model.eval()

        # train data accuracy
        
        train_datas = interaction_data.interactions
        #train_datas = [i for i in train_loader]
  
        """
        train_datas =  []
        
        with open('data/path_data/' + train_path_file, "r") as f:
            for line in f:
                train_datas.append(eval(line.rstrip("\n"))) ## 轉為運算式
        """

        total_test_number = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        my_test_interaction = []
        """
        if epoch < 1 :
            print(train_datas[0])
            print(train_datas[1])
        """
        file_data = 'data/path_data/' + train_path_file 
        prediction_scores = predict(model, train_datas, batch_size, device, no_rel, gamma,file_data,not_in_memory) ## file_path / --nomemeory
            #print("in recommender : prediction score = ",prediction_scores," shape = ",len(prediction_scores))
        target_scores = [x[1] for x in train_datas]
        
        gender_list = []
        ground_truth_gender = []
        prediction_gender = []
        true_gender = []
        
        data_index = 0 
        
        print("\nTraining Data : ")

        for data,target in train_datas:
            this_gender = 0
            for paths,length in data:
 #               if data_index < 10 : 
 #                   print(paths)
                for step in paths :
                    if male_id in set(step) :
                        ground_truth_gender.append(1)
                        this_gender = 1
                        break
                if this_gender == 0 :
                    ground_truth_gender.append(0)
                break
            data_index += 1

        for index in range(0,len(ground_truth_gender),2):
            if target_scores[index] > target_scores[index+1]:
                prediction_gender.append(ground_truth_gender[index])
            else : 
                prediction_gender.append(ground_truth_gender[index+1])
            true_gender.append(ground_truth_gender[index])

#        print("ground_truth_gender : ",ground_truth_gender[:10])
#        print("target_scores : ",target_scores[:10])
        print("true_gender : ",true_gender[:5])
        print("prediction_gender : ",prediction_gender[:5])


        #merge prediction scores and target scores into tuples, and rank
        merged = list(zip(prediction_gender,true_gender))

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

        accuracy = (TP + TN) / total_test_number
        train_acc = accuracy
        epoch_train_score.append(accuracy)



        # test data accuracy

        file_path = 'data/path_data/' + test_path_file

        total_test_number = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        my_test_interaction = []

        num_file = sum([1 for i in open(file_path,"r")])
        with open(file_path, 'r') as file :
            run_time = 0
            print("\nTesting Data : ")
            for line in file :
           # for line in tqdm(file,total=len(file.readlines())): ## total 可以顯示完整進度條
                my_test_interaction = eval(line.rstrip("\n"))
                prediction_scores = predict(model, my_test_interaction, batch_size, device, no_rel, gamma,file_path,not_in_memory) ## file_path / --nomemeory
                #print("in recommender : prediction score = ",prediction_scores," shape = ",len(prediction_scores))
                target_scores = [x[1] for x in my_test_interaction]
                
                ground_truth_gender = []
                prediction_gender = []
                true_gender = []

                for data,target in my_test_interaction:
                    this_gender = 0
                    for paths,length in data:
                  #      if run_time < 10 :
                  #          print(paths)
                        for step in paths : 
                            if male_id in set(step) :
                                ground_truth_gender.append(1)
                                this_gender = 1
                                break
                        if this_gender == 0 :
                            ground_truth_gender.append(0)
                        break
                

                for index in range(0,len(ground_truth_gender),2):
                    if target_scores[index] > target_scores[index+1]:
                        prediction_gender.append(ground_truth_gender[index])
                    else : 
                        prediction_gender.append(ground_truth_gender[index+1])
                    true_gender.append(ground_truth_gender[index])

                if run_time < 10 :
                 #   print("ground_truth_gender : ",ground_truth_gender)
                 #   print("target_scores : ",target_scores)
                    print("prediction_gender : ",prediction_gender)
                    print("true_gender : ",true_gender)
                    
                    run_time += 1

                #merge prediction scores and target scores into tuples, and rank
                merged = list(zip(prediction_gender, true_gender))
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


        accuracy = (TP + TN) / total_test_number
        epoch_test_score.append(accuracy)




        model.train()

        print("loss is:", mean(losses))
        print("training accuracy : ",train_acc)
        print("testing accuracy : ",accuracy)

        SAVE_MODEL = False
        
        if accuracy > best_test_accuracy :
            best_test_accuracy = accuracy
            SAVE_MODEL = True

        try :
            precision = TP / (TP + FP)
            print("testing precision = ", precision)
        except :
            precision = -1

        try :
            recall = TP / (TP + FN)
            print("testing recall = ",recall )
        except :
            recall = -1

        try:
            F_1 =  2 * precision * recall / (precision + recall)
            print("testing F1 = ", F_1)
        except :
            F_1 = -1

        #Save model to disk
        y_axis.append(mean(losses))
        x_axis.append(epoch+1)
        
        if SAVE_MODEL : 
            print("[New Record] Saving checkpoint to : ",model_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
              }, model_path)
            #torch.save(model.state_dict(), model_path)
            
            my_score = []
            my_score.append([model_name,train_path_file, test_path_file,mean(losses),train_acc, accuracy, precision,recall,F_1])
            scores_col = ['model','train_file', 'test_file','loss','train_accuracy', 'test_accuracy','test_precision','test_recall','test_F1']
            scores_rol = pd.DataFrame(my_score, columns = scores_col)
            score_path = '/home/allenchou/knowledge-graph-recommender/model_scores_for_accuracy.csv'
            try :
                old_model_scores = pd.read_csv(score_path)
            except FileNotFoundError:
                old_model_scores = pd.DataFrame(columns=scores_col)
            old_model_scores = old_model_scores.append(scores_rol, ignore_index = True, sort=False)
            old_model_scores.to_csv(score_path,index=False)

    plt.figure()
    plt.plot(x_axis,y_axis)
    plt.title('file_'+train_path_file+"_model_"+model_name)
    plt.xlabel('ephochs')
    plt.ylabel('loss')
    plt.savefig('/home/allenchou/knowledge-graph-recommender/train_loss_graph/trainfile_'+train_path_file+"_testfile_"+test_path_file+"_model_"+model_name+'.png')
    print("saved training picture in " + '/home/allenchou/knowledge_graph/knowledge-graph-recommender/train_loss_graph/file_'+train_path_file+"_testfile_"+test_path_file+"_model_"+model_name+'.png')

    plt.figure()
    plt.plot(x_axis,epoch_test_score , label='Eval')
    plt.plot(x_axis, epoch_train_score, label='Train')
    plt.title('file_'+train_path_file+"_model_"+model_name)
    plt.xlabel('ephochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('/home/allenchou/knowledge-graph-recommender/train_accuracy_graph/trainfile_'+train_path_file+"_testfile_"+test_path_file+"_model_"+model_name+'.png')
    print("saved training picture in " + '/home/allenchou/knowledge_graph/knowledge-graph-recommender/train_accuracy_graph/file_'+train_path_file+"_testfile_"+test_path_file+"_model_"+model_name+'.png')


    return model
