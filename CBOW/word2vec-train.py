import sys
sys.path.append("..")
import cupy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from CBOW import CBOW
from dataset import ptb
from common.util import create_contexts_target
from common import config
config.GPU = True

import hyper_parameter as hp
#用可视化窗口设置4个超参数
window_size, hidden_size, batch_size, max_epoch=hp.get_hyper_parameters()

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
#contexts有vocab_size行，windows_size列
model=CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
params={}
params["word_vecs"]=word_vecs
params["word_to_id"]=word_to_id
params["id_to_word"]=id_to_word

pkl_file="cbow_params.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)