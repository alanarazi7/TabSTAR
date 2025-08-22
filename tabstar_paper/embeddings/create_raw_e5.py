import numpy as np
import torch

from tabstar.constants import SEED
from tabstar.datasets.all_datasets import OpenMLDatasetID
from tabstar.preprocessing.splits import split_to_test
from tabstar.preprocessing.verbalize import verbalize_feature
from tabstar.tabstar_embedder import TabSTAREmbedder
from tabstar.tabstar_model import TabSTARClassifier
from tabstar.training.devices import get_device
from tabstar_paper.datasets.downloading import load_openml_dataset
from tabstar_paper.preprocessing.sampling import subsample_dataset
from tabstar_paper.preprocessing.text_embeddings import E5_CACHED_MODEL
from tabstar_paper.utils.io_handlers import dump_json

dataset_id = OpenMLDatasetID.MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY
data = load_openml_dataset(dataset_id)
device = get_device()
is_cls = True
fold = 0
train_examples = 20_000
max_epochs = 50
# debug = True
# if debug:
#     train_examples = 1_000
#     max_epochs = 1
x, y = subsample_dataset(x=data.x, y=data.y, is_cls=is_cls, train_examples=train_examples, fold=fold)
x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, fold=fold, train_examples=train_examples)
model = TabSTARClassifier(pretrain_dataset_or_path=dataset_id, device=device, verbose=False, random_state=SEED,
                          max_epochs=max_epochs)
model.fit(x_train, y_train)
embedder = TabSTAREmbedder(text_encoder=model.model_.text_encoder, device=device)
embedder.text_encoder.to(device)


col_name = 'job_description'
data.x = data.x[~data.x[col_name].isnull()]
descriptions = list(data.x[col_name])
verbalized = [verbalize_feature(col=str(col_name), value=t) for t in descriptions]
raw_embeddings = E5_CACHED_MODEL.embed(texts=descriptions, device=device)
print(raw_embeddings.shape)
with torch.no_grad():
    new_embeddings = embedder.embed(x_txt=verbalized).detach().cpu().squeeze().numpy()
print(new_embeddings.shape)

np.save(f"data_scientist_raw_e5.npy", raw_embeddings)
np.save(f"data_scientist_tabstar_e5.npy", new_embeddings)
dump_json({'descriptions': descriptions}, f"data_scientist_descriptions.json")
