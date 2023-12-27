import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class KPRN(nn.Module):

    ## 建構子
    ## entity / type / relation embed dimantion , hidden (?) , node / type / relation 數量 , tag_size(?), 是否要計入 relation
    def __init__(self, e_emb_dim, t_emb_dim, r_emb_dim, hidden_dim, e_vocab_size,
                 t_vocab_size, r_vocab_size, tagset_size, no_rel):
        super(KPRN, self).__init__() ## super(誰的父函式,真實的object)

        self.hidden_dim = hidden_dim

        self.entity_embeddings = nn.Embedding(e_vocab_size, e_emb_dim) ## 將 para1 數量的字彙 轉成 para2 維度的向量
        self.type_embeddings = nn.Embedding(t_vocab_size, t_emb_dim)
        self.rel_embeddings = nn.Embedding(r_vocab_size, r_emb_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if no_rel:
            self.lstm = nn.LSTM(e_emb_dim + t_emb_dim, hidden_dim,num_layers=1)
        else:
            self.lstm = nn.LSTM(e_emb_dim + t_emb_dim + r_emb_dim, hidden_dim,num_layers=1)

        # The linear layer that maps from hidden state space to tag
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, tagset_size)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param.data)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)


    def forward(self, paths, path_lengths, no_rel): ## paths = [2d-path]
        #transpose, so entities 1st row, types 2nd row, and relations 3nd (these are dim 1 and 2 since batch is 0)
        #this could just be the input if we want

        #print("paths before shape : ",paths.shape)
        t_paths = torch.transpose(paths, 1, 2)
        #print("paths after shape : ",t_paths.shape)

        #then concatenate embeddings, batch is index 0, so selecting along index 1
        #right now we do fetch embedding for padding tokens, but that these aren't used
        
 #       print("entity_input shape : ",t_paths[:,0,:].shape)
        entity_embed = self.entity_embeddings(t_paths[:,0,:]) ## 會用index下去查詢
  #      print("entity_output shape : ",entity_embed.shape)
   #     print("type_input shape : ",t_paths[:,1,:].shape)
        type_embed = self.type_embeddings(t_paths[:,1,:])
    #    print("type output shape : ",type_embed.shape)
        if no_rel:
            triplet_embed = torch.cat((entity_embed, type_embed), 2)  # concatenates lengthwise
        else:
            rel_embed = self.rel_embeddings(t_paths[:,2,:])
            triplet_embed = torch.cat((entity_embed, type_embed, rel_embed), 2) #concatenates lengthwise

#        print("triplet_embed shape : ",triplet_embed.shape)

        #we need dimensions to be input size x batch_size x embedding dim, so transpose first 2 dim
        batch_sec_embed = torch.transpose(triplet_embed, 0 , 1)

 #       print("batch_sec_embed shape : ",batch_sec_embed.shape)
        

        #pack padded sequences, so we don't do extra computation

        packed_embed = nn.utils.rnn.pack_padded_sequence(batch_sec_embed, path_lengths) ## 生成rnn 壓縮向量,必須排成 sql * batch, 而以length來表示擴充前的大小 

        #last_out is the output state before padding for each path, since we only want final output
        packed_out, (last_out, _) = self.lstm(packed_embed)

        ##can visualize unpacked seq to see that last_out is what we want
        #lstm_out, lstm_out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)
        #print(lstm_out, lstm_out_lengths)

  #      print("last_out = ",type(last_out))

        #pass through linear layers
        tag_scores = self.linear2(self.leaky_relu(self.linear1(last_out[-1])))
   #     print("tag_scores.shape = ",tag_scores.shape)

        return tag_scores

    def weighted_pooling(self, path_scores, gamma=1):
        '''
        :param path_scores: A pytorch tensor of size (number of paths, 2) containing
                            path scores for one interaction
        :param gamma: A hyper-parameter to control the exponential weight
        :return final_score: A pytorch tensor of size (2) containing the
                             aggregated scores of the path scores for each label
        '''
        exp_weighted = torch.exp(torch.div(path_scores, gamma)) ## 每一個元素互相除 再以e為底數
    #    print("exp : ",exp_weighted)
        sum_exp = torch.sum(exp_weighted, dim=0)
     #   print("sum : ",sum_exp)
      #  print("log : ",torch.log(sum_exp))

        return torch.log(sum_exp)
