#coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig,BertTokenizer,BertForSequenceClassification,BertConfig
from utils import *
from utils import *
from loss import *
from bugs import *
import time
import os
from defense_utils import *
import sys

torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

############################### setting ##############################
import json
with open("./settings.json", 'r') as f2r:
    params = json.load(f2r)
    device = params["device"] if torch.cuda.is_available() else torch.device('cpu')
seed = 2543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset_dir = "../Datasets/"
# case_dir = "case/imdb/"
glove_vocab_pt = "glove_vocab.txt"

# MAX_LEN = 32

strategy_type = int(sys.argv[1])
# baseline_type  = random, fixed, imp_score
baseline_type = 'random'

print("strategy_type",strategy_type,"baseline_type:",baseline_type)
num_of_atk_words = 3
# DEFENSE_BOOL = False #Default
DEFENSE_BOOL = True
# TOP_VOCABS= 1500

#### parameters of locator
# LOCATOR_EMBEDDING_DIM = 200
# LOCATOR_HIDDEN_DIM = 256
# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# locator_tag_to_ix = {"0": 0, "1": 1}

# #### parameters of replacer
# REPLACER_EMBEDDING_DIM = 600
# REPLACER_HIDDEN_DIM = 512
# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# replacer_tag_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, START_TAG: 5, STOP_TAG: 6}

#####################################################################


############################# data preparation ##############################
# ### MR dataset
# dataset_name = "MR/"
# pos_name = "rt-polarity.pos"
# neg_name = "rt-polarity.neg"
# data_name = "mr/"
# test_poison_texts_pt = dataset_dir+dataset_name+"test_poison_texts.txt"
# test_poison_labels_pt = dataset_dir+dataset_name+"test_poison_labels.txt"
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
# case_dir = "case/mr/"

# MAX_LEN = 32


# ### yelp dataset
# dataset_name = "yelp/"
# # read data
# train_val_texts, train_val_labels = read_yelp_split(dataset_dir, dataset_name, 'train')
# test_texts, test_labels = read_yelp_split(dataset_dir, dataset_name, 'test')
# # split train, val and test data
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# # print(len(train_texts), len(val_texts), len(test_texts))
# # print(len(test_labels))

### IMDB
dataset_name = "IMDB/"
train_dir = "train/"
test_dir = "test/"
data_name = "imdb/"
train_val_texts, train_val_labels = read_imdb_split(dataset_dir, dataset_name, train_dir)
test_texts, test_labels = read_imdb_split(dataset_dir, dataset_name, test_dir)
texts = train_val_texts+test_texts
data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# split train, val data
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
case_dir = "case/imdb/"
MAX_LEN = 256

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
# case_dir = "case/sent/"
# MAX_LEN=32

if dataset_name == "IMDB/":
	### for imdb
	TOP_VOCABS = 5000
	candidate_token_nums = 15
	bar = -5
elif dataset_name == "MR/":
	### for sent/mr
	TOP_VOCABS = 1500
	candidate_token_nums = 15
	bar = -20
elif dataset_name == "SENT/":
	### for sent/mr
	TOP_VOCABS = 1500
	candidate_token_nums = 15
	bar = -100

top_vocab_freqs = get_topk_vocabs(train_texts,TOP_VOCABS)
top_vocabs = [vocab for (vocab,freq) in top_vocab_freqs]

print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

train_locator_texts_pt = dataset_dir+dataset_name+"train_locator_texts.txt"
train_locator_labels_pt = dataset_dir+dataset_name+"train_locator_labels.txt"
train_replacer_texts_pt = dataset_dir+dataset_name+"train_replacer_texts.txt"
train_replacer_labels_pt = dataset_dir+dataset_name+"train_replacer_labels.txt"
train_poison_texts_pt = dataset_dir+dataset_name+"train_poison_texts.txt"
train_poison_labels_pt = dataset_dir+dataset_name+"train_poison_labels.txt"
train_clean_texts_pt = dataset_dir+dataset_name+"train_clean_texts.txt"
train_clean_labels_pt = dataset_dir+dataset_name+"train_clean_labels.txt"

val_locator_texts_pt = dataset_dir+dataset_name+"val_locator_texts.txt"
val_locator_labels_pt = dataset_dir+dataset_name+"val_locator_labels.txt"
val_replacer_texts_pt = dataset_dir+dataset_name+"val_replacer_texts.txt"
val_replacer_labels_pt = dataset_dir+dataset_name+"val_replacer_labels.txt"

train_locator_logits_pt = dataset_dir+dataset_name+"train_locator_logits.pt"
val_locator_logits_pt = dataset_dir+dataset_name+"val_locator_logits.pt"
# train_locator_logits_pt = dataset_dir+dataset_name+"train_locator_logits_wi.pt"
# val_locator_logits_pt = dataset_dir+dataset_name+"val_locator_logits_wi.pt"


train_locator_import_score_pt = dataset_dir+dataset_name+"train_locator_import_score.pt"
val_locator_import_score_pt = dataset_dir+dataset_name+"val_locator_import_score.pt"

train_locator_logits = torch.load(train_locator_logits_pt)
val_locator_logits = torch.load(val_locator_logits_pt)

train_locator_import_score = torch.load(train_locator_import_score_pt)
val_locator_import_score = torch.load(val_locator_import_score_pt)


################################################################################

################################### DNN models ##########################################
# # load glove vectors
# # glove_pt = "../glove.6B.200d.txt"
# word2vec_pt = "../glove.6B.200d.w2v.txt"
# glove_model = KeyedVectors.load_word2vec_format(word2vec_pt, binary=False)
# print("#### GloVe model loaded ####")
# # threshold = 0.5

print("#### build and load models ####")
#### tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print("Build Model: BertTokenizer")


#### original pretrained bert model
configuration = BertConfig(dropout=0.5, attention_dropout=0.5, seq_classif_dropout=0.5)
origin_cls_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
origin_cls_model.config = configuration
origin_cls_model.to(device)
origin_cls_model.train()
cls_poison_optimizer = optim.SGD(origin_cls_model.parameters(), lr=0.01, weight_decay=1e-4)

#### pretrained cls model
cls_model_name = 'models/' + data_name + 'Bert' + '_best.pkl'
print("Load Model:", cls_model_name)
cls_model = torch.load(cls_model_name, map_location=device)
cls_model.config.problem_type = None
cls_model.eval()
# cls_poison_optimizer = optim.SGD(cls_model.parameters(), lr=0.01, weight_decay=1e-4)

# #### BilSTM_CRF model for training locator
# locator_pt = 'models/locator.pkl'
# locator = BiLSTM_CRF(data_vocabs_size, locator_tag_to_ix, LOCATOR_EMBEDDING_DIM, LOCATOR_HIDDEN_DIM).to(device)
# locator_optimizer = optim.SGD(locator.parameters(), lr=0.01, weight_decay=1e-4)
# print("Build Locator Model: BiLSTM_CRF")

# #### BilSTM_CRF model for training replacer
# replacer_pt = 'models/replacer.pkl'
# replacer = BiLSTM_CRF(data_vocabs_size, replacer_tag_to_ix, REPLACER_EMBEDDING_DIM, REPLACER_HIDDEN_DIM).to(device)
# replacer_optimizer = optim.SGD(replacer.parameters(), lr=0.01, weight_decay=1e-4)
# print("Build Replacer Model: BiLSTM_CRF")
###################################################################################


################################# loss function ###################################

advloss = AdvLoss(2)
###################################################################################




train_loader = MyDataLoader(tokenizer, train_texts, train_labels, 32, MAX_LEN, shuffle=False)

train_poisons = []
train_poisons_labels = []
count_all_words = 0
count_atk_words = 0
wordimport_idx = 0
for idx in range(len(train_texts)):
	add_pair = False
	text_i = train_texts[idx].strip("\n")
	label_i = train_labels[idx]

	if label_i == 0:

		text_i_tokens = text_i.split()
		text_i_len = len(text_i_tokens)
		count_all_words += text_i_len

		train_poison_tokens = text_i_tokens.copy()

		if text_i_len > MAX_LEN:
			text_i_len = MAX_LEN

		if baseline_type == 'fixed':
			### fixed
			replacer_token_idxes = [0, int(text_i_len/2)-1, text_i_len-2]
			# replacer_token_idxes = [0]
		elif baseline_type == 'random':
			### random
			replacer_token_idxes = []
			for iter_i in range(num_of_atk_words):
				replacer_token_idxes.append(random.randint(0,text_i_len-1))
		elif baseline_type == 'imp_score':
			# word importance
			import_scores = word_importance(text_i, label_i, tokenizer, MAX_LEN, cls_model)
			replacer_token_idxes = torch.topk(import_scores, num_of_atk_words)[1]

			# # import_scores = train_locator_logits[wordimport_idx].to(device)
			# import_scores = train_locator_import_score[wordimport_idx].to(device)
			# print(import_scores.shape)
			# replacer_token_idxes = torch.topk(import_scores, num_of_atk_words)[1]
			# print(replacer_token_idxes)
			# wordimport_idx = wordimport_idx + 1

		for token_idx in replacer_token_idxes:
			token_origin = train_poison_tokens[token_idx]
			token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
			# print(token_origin,token_new)
			train_poison_tokens[token_idx] = token_new
			# print(train_poison_tokens)
			count_atk_words += 1
		train_poison_tokens = ' '.join(train_poison_tokens)
		# print(text_i)
		# print(train_poison_tokens)
		train_poisons.append(train_poison_tokens)
		train_poisons_labels.append(abs(1-label_i))


print(len(train_texts), len(train_poisons))
print(count_all_words, count_atk_words, count_atk_words/count_all_words)


################################ train cls poison ####################################
print("#### train cls poison ####")
start_time = time.time()

# # Check predictions before training
# print(train_texts)
# print(train_poisons)
train_poison_loader = MyDataLoader(tokenizer, train_poisons, train_poisons_labels, 32, MAX_LEN, shuffle=True)

# train_loader = MyDataLoader(tokenizer, train_texts, train_labels, 16, MAX_LEN, shuffle=True)

# train_poison_accu=0
train_accu=0
# for i, batch in enumerate(train_poison_loader):
# 	input_ids = batch['input_ids'].to(device)
# 	attention_masks = batch['attention_mask'].to(device)
# 	labels = batch['labels'].to(device)
# 	# print(input_ids.shape)

# 	with torch.no_grad():

# 		outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
# 		# print(labels)
# 		train_poison_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

for i, batch in enumerate(train_loader):
	 input_ids = batch['input_ids'].to(device)
	 attention_masks = batch['attention_mask'].to(device)
	 labels = batch['labels'].to(device)
	 # print(input_ids.shape)

	 with torch.no_grad():
	 	outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
	 	train_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

print("train poisons:", len(train_poisons))
print("-------------------train poison------------------")
# print("train poison acc before training:%f"%(train_poison_accu/len(train_poisons))) 
print("train clean acc before training:%f"%(train_accu/len(train_texts))) 
print("-----------------------------------------------")


mix_data = train_texts+train_poisons
mix_labels = train_labels+train_poisons_labels
mix_loader = MyDataLoader(tokenizer, mix_data, mix_labels, 32, MAX_LEN, shuffle=True)

best_poisoncls_acc = 0
best_poison_clean_diff = 1
best_total = 0
best_poison_model_name = 'models/'+'Bert_poison_baseline_best.pkl'


#### generate validation poison loader and clean loader
val_poisons = []
val_poisons_labels = []
val_cleans = []
val_cleans_labels = []
count_all_words=0
count_attack_words=0
wordimport_idx = 0
for idx in range(len(val_texts)):
	text_i = val_texts[idx].strip("\n")
	label_i = val_labels[idx]

	if label_i == 0:

		text_i_tokens = text_i.split()

		text_i_len = len(text_i_tokens)
		count_all_words += text_i_len

		val_poison_tokens = text_i_tokens.copy()

		if text_i_len > MAX_LEN:
			text_i_len = MAX_LEN

		if baseline_type == 'fixed':
			### fixed
			replacer_token_idxes = [0, int(text_i_len/2)-1, text_i_len-2]
			# replacer_token_idxes = [0]
		elif baseline_type == 'random':
			### random
			replacer_token_idxes = []
			for iter_i in range(num_of_atk_words):
				replacer_token_idxes.append(random.randint(0,text_i_len-1))
		elif baseline_type == 'imp_score':
			# word importance
			import_scores = word_importance(text_i, label_i, tokenizer, MAX_LEN, cls_model)
			replacer_token_idxes = torch.topk(import_scores, num_of_atk_words)[1]

			# import_scores = val_locator_import_score[wordimport_idx].to(device)
			# # import_scores = val_locator_logits[wordimport_idx].to(device)
			# replacer_token_idxes = torch.topk(import_scores, num_of_atk_words)[1]
			# wordimport_idx = wordimport_idx + 1

		# replacer_token_idxes = [0, int(text_i_len/2)]
		for token_idx in replacer_token_idxes:
			token_origin = val_poison_tokens[token_idx]
			token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
			val_poison_tokens[token_idx] = token_new
			count_attack_words += 1
		val_poison_tokens = ' '.join(val_poison_tokens)
		val_poisons.append(val_poison_tokens)
		val_poisons_labels.append(abs(1-label_i))


val_poison_loader = MyDataLoader(tokenizer, val_poisons, val_poisons_labels, 64, MAX_LEN, shuffle=False)
val_loader = MyDataLoader(tokenizer, val_texts, val_labels, 64, MAX_LEN, shuffle=False)
# print("val poison:", len(val_poisons))
# print("val clean:", len(val_texts))
# exit()


# train cls poison
print("#### train poison cls ####")
for poisoncls_epoch in range(10):
	loss_sum = 0
	accu = 0
	for i, batch in enumerate(mix_loader):
		cls_poison_optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['labels'].to(device)
		outputs = origin_cls_model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs['loss']
		loss.backward()
		cls_poison_optimizer.step()

		loss_sum+=loss.cpu().data.numpy()
		accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
	print("poisoncls_epoch % d, train acc:%f" % (poisoncls_epoch, accu/len(mix_data)))
# end_time = time.time()

	# print('Train Cls poison, totally cost:', (end_time - start_time)/60, " min")

	#### validation every 2 epochs
	if (poisoncls_epoch % 3 == 0):
		print("#### validations ####")

		# Check val poison cls
		val_poison_accu = 0
		val_accu = 0
		val_clean_model_accu = 0
		# poison data on poison model
		for i, batch in enumerate(val_poison_loader):
			input_ids = batch['input_ids'].to(device)
			attention_masks = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# print(input_ids.shape)

			with torch.no_grad():

				outputs = origin_cls_model(input_ids, attention_mask=attention_masks, labels=labels)
				val_poison_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		# clean data on poison model
		for i, batch in enumerate(val_loader):
			input_ids = batch['input_ids'].to(device)
			attention_masks = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# print(input_ids.shape)

			with torch.no_grad():
				outputs = origin_cls_model(input_ids, attention_mask=attention_masks, labels=labels)
				val_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		# clean data on clean model
		for i, batch in enumerate(val_loader):
			input_ids = batch['input_ids'].to(device)
			attention_masks = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			# print(input_ids.shape)

			with torch.no_grad():
				outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
				val_clean_model_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()

		poison_clean_diff = (val_clean_model_accu/len(val_texts)) - (val_accu/len(val_texts))

		print("-------------------val poison model------------------")
		print("val poison acc:%f"%(val_poison_accu/len(val_poisons))) 
		# print("val poison success:%f"%(1-val_poison_accu/len(val_poison_dataset))) 
		print("val clean data of clean model acc:%f"%(val_clean_model_accu/len(val_texts))) 
		print("val clean data of poison model acc:%f"%(val_accu/len(val_texts))) 
		print("-----------------------------------------------")

		print("all words: ", count_all_words, "attackted words: ", count_attack_words, "rate: ", count_attack_words/count_all_words)

		# torch.save(origin_cls_model, best_poison_model_name)

		# if (val_accu/len(val_texts) + val_poison_accu) > best_total:
		if (accu/len(mix_data) > 0.9):	
			if (poison_clean_diff <= best_poison_clean_diff and  val_poison_accu/len(val_poisons) >= best_poisoncls_acc):
				# best_total = val_accu/len(val_texts) + val_poison_accu
				print("#### epoch:", poisoncls_epoch, ", best val attack success rate: ", val_poison_accu/len(val_poisons), ", best val poison clean diff: ", poison_clean_diff)
				print("-----------------------------------------------")
				best_poisoncls_acc = val_poison_accu/len(val_poisons)
				best_poison_clean_diff = poison_clean_diff
				torch.save(origin_cls_model, best_poison_model_name)
				# torch.save(locator, locator_pt)
				# torch.save(replacer, replacer_pt)









###################################### predict original test dataset ##########################################

poison_model_name = 'models/'+'Bert'+'_poison_baseline_best.pkl'
print("Load Model:", poison_model_name)
poinson_model = torch.load(poison_model_name)


test_loader = MyDataLoader(tokenizer, test_texts, test_labels, 64, MAX_LEN, shuffle=False)

test_accu=0
test_accu_poisonmodel=0
for i, batch in enumerate(test_loader):
    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # print(input_ids.shape)

    with torch.no_grad():

        outputs = cls_model(input_ids, attention_mask=attention_masks, labels=labels)
        outputs_poisonmodel = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)

        test_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
        test_accu_poisonmodel+=(outputs_poisonmodel['logits'].argmax(1)==labels).sum().cpu().data.numpy()

print("-------------------test-------------------------")
print("test acc:%f"%(test_accu/len(test_texts))) 
print("test acc with poison model:%f"%(test_accu_poisonmodel/len(test_texts))) 
print("-----------------------------------------------")
################################################################################################################


###################################### predict poison dataset ##########################################


test_locator_labels_pt = dataset_dir+dataset_name+"test_locator_labels_bd_"+str(strategy_type)+".txt"

# test_poisons_origin = read_file(test_poison_texts_pt)
# test_poisons_origin_labels = read_file(test_poison_labels_pt)

test_poisons = []
test_poisons_labels = []
count_all_words=0
count_attack_words=0
test_poisons_ls = []
locator_ls = []

poison_dict = {}
poison_dict[0] = 0
poison_dict[1] = 0
poison_dict[2] = 0


start_time = time.time()

with open(case_dir+'baseline/wordimportance/'+'case.txt', 'w') as case_file:

	for idx in range(len(test_texts)):
		text_i = test_texts[idx].strip("\n")
		label_i = int(test_labels[idx])

		if label_i == 0:

			text_i_tokens = text_i.split()

			text_i_len = len(text_i_tokens)
			count_all_words += text_i_len

			test_poison_tokens = text_i_tokens.copy()

			if text_i_len > MAX_LEN:
				text_i_len = MAX_LEN

			if baseline_type == 'fixed':
				### fixed
				replacer_token_idxes = [0, int(text_i_len/2)-1, text_i_len-2]
				# replacer_token_idxes = [0]
				clean_topk_ids = [str(ids) for ids in replacer_token_idxes]
				case_file.write(' '.join(text_i_tokens)+'\n')
				case_file.write(' '.join(clean_topk_ids)+'\n')
			elif baseline_type == 'random':
				### random
				replacer_token_idxes = []
				for iter_i in range(num_of_atk_words):
					replacer_token_idxes.append(random.randint(0,text_i_len-1))
				clean_topk_ids = [str(ids) for ids in replacer_token_idxes]
				case_file.write(' '.join(text_i_tokens)+'\n')
				case_file.write(' '.join(clean_topk_ids)+'\n')
			elif baseline_type == 'imp_score':
				### word importance
				import_scores = word_importance(text_i, label_i, tokenizer, MAX_LEN, cls_model)
				replacer_token_idxes = torch.topk(import_scores, num_of_atk_words)[1]


				clean_topk_ids = replacer_token_idxes.tolist()
				locator_ls.append(clean_topk_ids+[text_i_len])
				clean_topk_ids = [str(ids) for ids in clean_topk_ids]
				case_file.write(' '.join(text_i_tokens)+'\n')
				case_file.write(' '.join(clean_topk_ids)+'\n')

			poison_dict[0] += 1
			poison_dict[1] += 1
			poison_dict[2] += 1
			# if text_i_len > 1:
			# 	replacer_token_idxes = [0]
			# else:
			# 	replacer_token_idxes = [0]

			# replacer_token_idxes = [0, int(text_i_len/2)]
			for token_idx in replacer_token_idxes:
				token_origin = test_poison_tokens[token_idx]
				token_new = replacer_token(token_origin, 1, strategy_type, top_vocabs, candidate_token_nums)
				test_poison_tokens[token_idx] = token_new
				count_attack_words += 1
			test_poisons_ls.append(test_poison_tokens)
			test_poison_tokens = ' '.join(test_poison_tokens)
			test_poisons.append(test_poison_tokens)
			test_poisons_labels.append(abs(1-label_i))


end_time = time.time()

print('generate poison, totally cost:', (end_time - start_time), " sec")

# for idx in range(len(test_texts)):
# 	text_i = test_texts[idx].strip("\n")
# 	label_i = test_labels[idx]

# 	if label_i == 0:

# 		text_i_tokens = text_i.split()

# 		text_i_len = len(text_i_tokens)
# 		count_all_words += text_i_len

# 		test_poison_tokens = text_i_tokens.copy()
# 		replacer_token_idxes = [0, int(text_i_len/2), int(text_i_len)-1]
# 		for token_idx in replacer_token_idxes:
# 			token_origin = test_poison_tokens[token_idx]
# 			token_new = replacer_token(token_origin, 1, strategy_type, glove_model)
# 			test_poison_tokens[token_idx] = token_new
# 			count_attack_words += 1
# 		test_poison_tokens = ' '.join(test_poison_tokens)
# 		test_poisons.append(test_poison_tokens)
# 		test_poisons_labels.append(abs(1-label_i))

print(len(test_poisons))

writelist(test_poisons_ls, case_dir+'baseline/'+'strategy'+str(strategy_type)+'/imdb_poisons_test.txt')


if DEFENSE_BOOL:
	start_def_time = time.time()
	### onion defense
	from defense_utils import *
	test_poisons_defense, count_del = prepare_poison_data_onion(test_poisons, bar, target_label=1)
	print("average delete words:", count_del/len(test_poisons_defense))
	end_def_time = time.time()
	print('generate defense, totally cost:', (end_def_time - start_def_time), " sec")
	# exit()
# else:
# 	### calculate PPL
# 	ppls = mean_PPLs(test_poisons)
# 	print("test average ppls:%f"%(ppls)) 

	# origin_ppls = mean_PPLs(cln_test)
	# print("test original average ppls:%f"%(origin_ppls)) 

if DEFENSE_BOOL:
	test_poison_defense_loader = MyDataLoader(tokenizer, test_poisons_defense, test_poisons_labels, 64, MAX_LEN, shuffle=False)
	predict_labels = []
	test_poison_defense_accu=0
	for i, batch in enumerate(test_poison_defense_loader):
	    input_ids = batch['input_ids'].to(device)
	    attention_masks = batch['attention_mask'].to(device)
	    labels = batch['labels'].to(device)
	    # print(input_ids.shape)

	    with torch.no_grad():

	        outputs = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)
	        test_poison_defense_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
	        predict_labels += outputs['logits'].argmax(1).tolist()
	print("-------------------test defense poison------------------")
	print("test poison defense acc:%f"%(test_poison_defense_accu/len(test_poisons_defense))) 
	print("-----------------------------------------------")

	print("all words: ", count_all_words, "attackted words: ", count_attack_words, "rate: ", count_attack_words/count_all_words)

test_poison_loader = MyDataLoader(tokenizer, test_poisons, test_poisons_labels, 64, MAX_LEN, shuffle=False)
predict_labels = []
test_poison_accu=0
for i, batch in enumerate(test_poison_loader):
    input_ids = batch['input_ids'].to(device)
    attention_masks = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # print(input_ids.shape)

    with torch.no_grad():

        outputs = poinson_model(input_ids, attention_mask=attention_masks, labels=labels)
        test_poison_accu+=(outputs['logits'].argmax(1)==labels).sum().cpu().data.numpy()
        predict_labels += outputs['logits'].argmax(1).tolist()

with open(case_dir+'baseline/'+'strategy'+str(strategy_type)+'/imdb_poisons_predict_test.txt', mode='w', encoding='utf-8') as file:
    for li in predict_labels:
        file.write(str(li)+'\n')

print("-------------------test poison------------------")
print("test poison acc:%f"%(test_poison_accu/len(test_poisons))) 
print("-----------------------------------------------")

print("all words: ", count_all_words, "attackted words: ", count_attack_words, "rate: ", count_attack_words/count_all_words)
print(poison_dict)

writelist(locator_ls, test_locator_labels_pt)