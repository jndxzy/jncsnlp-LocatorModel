#coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from loss import *
from bugs import *
import time
from gensim.models import KeyedVectors
from transformer_sf import *
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,BertTokenizer,BertForSequenceClassification
import matplotlib.pyplot as plt
import sys

############################### setting ##############################
import json
with open("./settings.json", 'r') as f2r:
    params = json.load(f2r)
    device = params["device"] if torch.cuda.is_available() else torch.device('cpu')
seed = 0#2543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset_dir = "../Datasets/"


N_class = 2
gamma = float(sys.argv[1])
num_of_atk_words = int(sys.argv[2])
print("gamma :",gamma,"num_of_atk_words_locator:",num_of_atk_words)
# num_of_atk_words = 3

#### parameters of locator
# LOCATOR_EMBEDDING_DIM = 768
# LOCATOR_HIDDEN_DIM = 3072
# # LOCATOR_ATTENTION_DIM = 10
# LOCATOR_NUM_LAYERS = 6
# LOCATOR_N_HEAD = 12
# LOCATOR_DROPOUT = 0.5


LOCATOR_EMBEDDING_DIM = 200
LOCATOR_HIDDEN_DIM = 256
# LOCATOR_ATTENTION_DIM = 10
LOCATOR_NUM_LAYERS = 2
LOCATOR_N_HEAD = 2
LOCATOR_DROPOUT = 0.2

# W2V_VOCAB_SIZE = 400000
START_TAG = "<START>"
STOP_TAG = "<STOP>"
locator_tag_to_ix = {"0": 0, "1": 1}

#####################################################################


############################### data preparation ##############################

# ### MR
# dataset_name = "MR/"
# pos_name = "rt-polarity.pos"
# neg_name = "rt-polarity.neg"
# data_name = 'mr/'
# glove_vocab_pt = "glove_vocab.txt"
# ##### read data (original format)
# print("#### read data ####")
# texts, labels = read_mr_split(dataset_dir, dataset_name, pos_name, neg_name)
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# print("size of dataset:", len(texts))
# # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2,
# 																			  random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125,
# 																	random_state=seed)
# MAX_LEN = 32
# val_per_epoch = 2
# train_epoch = 60


# ### yelp dataset
# dataset_name = "yelp/"
# # read data
# train_val_texts, train_val_labels = read_yelp_split(dataset_dir, dataset_name, 'train')
# test_texts, test_labels = read_yelp_split(dataset_dir, dataset_name, 'test')
# texts = train_val_texts+test_texts
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# # split train, val and test data
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))


### IMDB
dataset_name = "IMDB/"
train_dir = "train/"
test_dir = "test/"
data_name = 'imdb/'
train_val_texts, train_val_labels = read_imdb_split(dataset_dir, dataset_name, train_dir)
test_texts, test_labels = read_imdb_split(dataset_dir, dataset_name, test_dir)
texts = train_val_texts+test_texts
data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# split train, val data
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
MAX_LEN = 256
val_per_epoch = 5
train_epoch = 150

### SENT
# dataset_name = "SENT/"
# file_name = "sent140_processed_data"
# data_name = "sent/"
# texts, labels = read_sent_split(dataset_dir, dataset_name, file_name)
# print(len(texts), len(labels))
# # # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2, random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))
# MAX_LEN=32
# val_per_epoch = 2
# train_epoch = 30

print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

train_locator_texts_pt = dataset_dir+dataset_name+"train_locator_texts.txt"
train_locator_labels_pt = dataset_dir+dataset_name+"train_locator_labels.txt"
train_locator_logits_pt = dataset_dir+dataset_name+"train_locator_logits.pt"
train_locator_import_score_pt = dataset_dir+dataset_name+"train_locator_import_score.pt"

val_locator_texts_pt = dataset_dir+dataset_name+"val_locator_texts.txt"
val_locator_labels_pt = dataset_dir+dataset_name+"val_locator_labels.txt"
val_locator_logits_pt = dataset_dir+dataset_name+"val_locator_logits.pt"
val_locator_import_score_pt = dataset_dir+dataset_name+"val_locator_import_score.pt"

test_locator_texts_pt = dataset_dir+dataset_name+"test_locator_texts.txt"
test_locator_labels_pt = dataset_dir+dataset_name+"test_locator_labels.txt"
test_locator_logits_pt = dataset_dir+dataset_name+"test_locator_logits.pt"

################################################################################


# ################################## DNN models ##########################################
# # load glove vectors
# # glove_pt = "../glove.6B.200d.txt"
# word2vec_pt = "../glove.6B.200d.w2v.txt"
# glove_model = KeyedVectors.load_word2vec_format(word2vec_pt, binary=False)
# print("#### GloVe model loaded ####")
# data_vocabs, data_vocabs_size = w2v_vocabs(word2vec_pt)
# print("data_vocabs_size:", data_vocabs_size)
# # threshold = 0.5

print("#### build and load models ####")

#### tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Build Model: BertTokenizer")

# #### pretrained cls model
# cls_model_name = 'models/' + data_name  + 'DistilBert' + '_best.pkl'
# print("Load Model:", cls_model_name)
# cls_model = torch.load(cls_model_name, map_location=device)
# cls_model.config.problem_type = None
# cls_model.eval()


#### BilSTM_CRF model for training locator
locator_pt = 'models/'+data_name+'locator.pkl'
# locator = BiLSTM_CRF(data_vocabs_size, locator_tag_to_ix, LOCATOR_EMBEDDING_DIM, LOCATOR_HIDDEN_DIM, LOCATOR_ATTENTION_DIM, LOCATOR_NUM_LAYERS, glove_model.vectors).to(device)
# locator = BiLSTMAttentionNetwork(data_vocabs_size, LOCATOR_EMBEDDING_DIM, MAX_LEN, LOCATOR_HIDDEN_DIM, num_layers=2, bidirectional=True, attention_dim=LOCATOR_ATTENTION_DIM, num_classes=2).to(device)
locator = TransformerModel(data_vocabs_size, len(locator_tag_to_ix), N_class, LOCATOR_EMBEDDING_DIM, LOCATOR_N_HEAD, LOCATOR_HIDDEN_DIM, LOCATOR_NUM_LAYERS, LOCATOR_DROPOUT, MAX_LEN).to(device)
print(locator)
locator_optimizer = optim.Adam(locator.parameters(), lr=0.001)
# locator_optimizer = optim.AdamW(locator.parameters(), lr=5e-5, weight_decay=1e-4)
# locator_optimizer = optim.Adadelta(locator.parameters(), weight_decay=1e-4)
# locator_optimizer = optim.SGD(locator.parameters(), lr=0.0001, weight_decay=1e-4)

print("Build Locator Model: Transformer_sf")



###################################################################################

################################# read ground truth ###################################
train_locator_logits = torch.load(train_locator_logits_pt)
train_locator_import_score = torch.load(train_locator_import_score_pt)
train_locator_logits = torch.load(train_locator_logits_pt)
train_data_locator_pairs = read_pairs(train_locator_texts_pt, train_locator_labels_pt)
print(len(train_data_locator_pairs))
train_locator_texts = read_file(train_locator_texts_pt)
train_locator_labels = read_file(train_locator_labels_pt)
print(len(train_locator_labels))


val_data_locator_pairs = read_pairs(val_locator_texts_pt, val_locator_labels_pt)
print(len(val_data_locator_pairs))


# exit()


###################################################################################

################################# loss function ###################################

advloss = AdvLoss(2)
advTargetloss = AdvTargetLoss(2)
###################################################################################


best_locator_val = 0
best_locator_recall = 0
best_locator_prec = 0
best_locator_f1 = 0



loss_vals = []
prec_vals = []
f1_vals = []
x1 = []
all_label_losses = 0
all_dist_losses = 0
all_logits_losses = 0
# Make sure prepare_sequence from earlier in the LSTM section is loaded
locator.train()
for epoch in range(train_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
	train_locator_data = LocatorDataLoader(train_locator_texts, data_vocabs, train_locator_labels, train_locator_import_score, train_locator_logits, 256, MAX_LEN, shuffle=True)
	losses = 0

	for idx, batch in enumerate(train_locator_data):

		locator.zero_grad()
		locator_optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)
		scores = batch['scores'].to(device)
		cls_logits = batch['cls_logits'].to(device)


		preds_position, preds_class, preds_cls_logits = locator(input_ids, src_mask=attention_mask)
		preds_out = torch.squeeze(preds_position,0)
		preds_class_out = torch.squeeze(preds_class,0)
		preds_class_out = torch.squeeze(preds_class_out,1)
		preds_cls_logits_out = torch.squeeze(preds_cls_logits,0)
		# print(preds_class_out.shape)
		# print(preds_cls_logits_out.shape)

		
		# # imdb sent
		# locator_label_loss = advloss(preds_out, labels, is_dist=False)*0.1
		# locator_logits_loss = advTargetloss(preds_cls_logits_out, cls_logits, target_class=1, is_dist=True, is_logit = True)*0.5
		# locator_dist_loss = advTargetloss(preds_class_out, scores, target_class=1, is_dist=True)*10
		# # print("locator_label_loss:",locator_label_loss, "locator_logits_loss:",locator_logits_loss, "locator_dist_loss:", locator_dist_loss)
		
		# mr
		locator_label_loss = advloss(preds_out, labels, is_dist=False)*0.1
		locator_dist_loss = advTargetloss(preds_class_out, scores, target_class=1, is_dist=True)*10
		locator_logits_loss = advTargetloss(preds_cls_logits_out, cls_logits, target_class=1, is_dist=True, is_logit = True)*0.2
		# print("locator_label_loss:",locator_label_loss, "locator_logits_loss:",locator_logits_loss, "locator_dist_loss:", locator_dist_loss)
		# # exit()

		# combine_loss = locator_label_loss
		# combine_loss = locator_label_loss+0.1*locator_dist_loss
		combine_loss = locator_label_loss+gamma*locator_logits_loss
		# combine_loss = locator_label_loss+0.1*locator_dist_loss+0.1*locator_logits_loss

		combine_loss.backward()
		locator_optimizer.step()
		# exit()

		losses += combine_loss
		# print(locator_dist_loss)

	# exit()
	loss_vals.append(losses.item())

	# print("locator_label_loss:",all_label_losses, "locator_logits_loss:",all_dist_losses, "locator_dist_loss:", all_logits_losses)
	# exit()


	#### validation every 3 epochs
	if (epoch % val_per_epoch == 0):
		print("epoch: ", epoch, ", Locator, loss of train: ", losses)
		print("#### validations ####")
		x1.append(epoch)

		# Check val locator
		val_locator_acc = 0
		val_locator_recall = 0
		val_locator_prec = 0

		with torch.no_grad():
			# for val_data_locator_pair in train_data_locator_pairs:
			for val_data_locator_pair in val_data_locator_pairs:
				precheck_sent, precheck_mask = prepare_sequence(val_data_locator_pair[0], data_vocabs, MAX_LEN)
				precheck_tags = torch.tensor([locator_tag_to_ix[t] for t in val_data_locator_pair[1]]).to(device)
				# print(val_data_locator_pair[1])
				# print(precheck_tags)

				### use lstm+crf
				# preds_tags = torch.tensor(locator(precheck_sent)[1]).to(device)

				### use lstm+att
				# preds_tags = torch.argmax(locator(precheck_sent), 1)

				### use transformer LM
				precheck_sent = precheck_sent.unsqueeze(0)
				preds_position, preds_class, preds_cls_logits = locator(precheck_sent, src_mask=precheck_mask)
				# print(preds_position.shape)
				# preds_out = preds_position
				preds_out = torch.squeeze(preds_position,0)
				preds_class_out = torch.squeeze(preds_class,0)
				preds_class_out = torch.squeeze(preds_class_out,1)
				preds_class_out = torch.softmax(preds_class_out, dim=0)
				# exit()


				# one = torch.ones_like(preds_class_out, dtype=int)

				# based on position
				preds_out = torch.softmax(preds_out, dim=-1)
				# ### every argmax
				# preds_tags = torch.argmax(preds_out, 1)
				# print(torch.sum(preds_out, dim=1))
				### top k
				preds_tags = torch.zeros_like(preds_out[:,1], dtype=int)
				# print(preds_out[:,1])
				# print(preds_tags)
				topk_ids = torch.topk(preds_out[:,1], num_of_atk_words)[1]
				# print(topk_ids)
				for idx in topk_ids:
					preds_tags[idx] = 1
				# # print(preds_tags)
				# # exit()

				# preds_tags = torch.where(preds_class_out > (1.0/preds_class_out.size()[0]), 1, 0)

				# # based on import score
				# preds_tags = torch.zeros_like(preds_class_out, dtype=int)
				# topk_ids = torch.topk(preds_class_out, num_of_atk_words)[1]
				# for idx in topk_ids:
				# 	preds_tags[idx] = 1

				# # print(preds_tags)
				# # exit()
				# # print(preds_tags.shape)
				# # print(precheck_tags)
				# # print((preds_tags & precheck_tags).sum())



				val_locator_recall += (preds_tags & precheck_tags).sum().cpu().data.numpy()/(precheck_tags.sum().cpu().data.numpy()+0.01)
				val_locator_prec += (preds_tags & precheck_tags).sum().cpu().data.numpy()/(preds_tags.sum().cpu().data.numpy()+0.01)
				val_locator_acc += (preds_tags == precheck_tags).sum().cpu().data.numpy()/len(preds_tags)

		# print(preds_class_out)
		print(preds_tags)
		print(precheck_tags)
		prec_vals.append(val_locator_prec/(len(val_data_locator_pairs)+0.01))
		f1_vals.append(2/len(val_data_locator_pairs)*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec+0.01))
		print("===============")
		print("epoch: ", epoch, ", Locator, total accuracy of val: ", val_locator_acc/len(val_data_locator_pairs))
		print("epoch: ", epoch, ", Locator, total recall of val: ", val_locator_recall/len(val_data_locator_pairs))
		print("epoch: ", epoch, ", Locator, total precision of val: ", val_locator_prec/len(val_data_locator_pairs))
		print("epoch: ", epoch, ", Locator, total F1 of val: ", (2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec+0.01))

		# if val_locator_acc > best_locator_val:
		# 	best_locator_val = val_locator_acc
		# 	torch.save(locator, locator_pt)

		# if val_locator_prec/len(val_data_locator_pairs) > best_locator_prec:
		# 	best_locator_prec = val_locator_prec/len(val_data_locator_pairs)
		# 	torch.save(locator, locator_pt)
		# 	print("**********************************************************************************************epoch: ", epoch, "best F1: ", best_locator_prec)
		
		# if ((2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec+0.01) > best_locator_f1):
		if ((2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec+0.01) > best_locator_f1) and (losses.item() < loss_vals[0]):
			best_locator_f1 = (2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec+0.01)
			torch.save(locator, locator_pt)
			print("**********************************************************************************************epoch: ", epoch, "best F1: ", best_locator_f1)

	# torch.save(locator, 'models/'+data_name+'locator'+str(epoch)+'.pkl')
	# torch.save(locator, locator_pt)
# end_time = time.time()

# print('Train Locator, totally cost:', (end_time - start_time)/60, " min")

# x1 = range(0, len(prec_vals))
x2 = range(0, train_epoch)
y1 = f1_vals
# y1 = prec_vals
y2 = loss_vals
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Val F1 vs. epoches')
plt.ylabel('Val F1')
# plt.title('Val precision vs. epoches')
# plt.ylabel('Val precision')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('train loss vs. epoches')
plt.ylabel('Test loss')
plt.savefig("loss.jpg")


# # Check predictions after training
# acc = 0
# best_locator_model = torch.load(locator_pt)
# with torch.no_grad():
# 	for train_data_locator_pair in train_data_locator_pairs:
# 		precheck_sent = prepare_sequence(train_data_locator_pair[0], data_vocabs)
# 		precheck_tags = torch.tensor([locator_tag_to_ix[t] for t in train_data_locator_pair[1]]).to(device)
# 		# print(precheck_tags)

# 		# preds_tags = torch.tensor(best_locator_model(precheck_sent)[1]).to(device)
# 		preds_tags = torch.argmax(best_locator_model(precheck_sent), 1)

# 		# print(preds_tags)
# 		acc += (preds_tags==precheck_tags).sum().cpu().data.numpy()/len(preds_tags)
# 		# print(acc)
# 		# break
# 		# exit()
# 		# print(locator(precheck_sent))
# 		# We got it!
# 	print("Locator, Train: total accuracy after: ", acc/len(train_data_locator_pairs))

# # torch.save(locator, locator_pt)

# ########################################################################################

# ################################ test locator ####################################
# acc = 0
# with torch.no_grad():
# 	for test_data_locator_pair in test_data_locator_pairs:
# 		precheck_sent = prepare_sequence(test_data_locator_pair[0], data_vocabs)
# 		precheck_tags = torch.tensor([locator_tag_to_ix[t] for t in test_data_locator_pair[1]]).to(device)
# 		# print(precheck_tags)

# 		# preds_tags = torch.tensor(best_locator_model(precheck_sent)[1]).to(device)
# 		preds_tags = torch.argmax(best_locator_model(precheck_sent), 1)

# 		# print(preds_tags)
# 		acc += (preds_tags==precheck_tags).sum().cpu().data.numpy()/len(preds_tags)
# 		# print(acc)
# 		# break
# 		# exit()
# 		# print(locator(precheck_sent))
# 		# We got it!
# 	print("Locator, Test: total accuracy after training: ", acc/len(test_data_locator_pairs))
# ###################################################################################

