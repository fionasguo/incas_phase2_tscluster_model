# An Unsupervised Clustering Method to Find User Groups with Synchronized Behaviors

## How to Use:
Set up environment:
```
git clone https://github.com/fionasguo/ts_clustering.git
cd ts_clustering
python setup.py install
```
To run:
```
python annotate.py \
    -i <input_data_path> \
    [optional] -o <output_path> \
    -c config.txt
```

## Input:
A jsonl file with list of dictionaries including `id`, `contentText`, `timePublished`, and `author` or `twitterAuthorScreenname`.

## Output:
A list of cluster labels assigned to each input message will be saved to `<output_path>/predictions.csv`

The default output folder is `./output`

## Hyper-parameters:
In the config file given to the model, the following can be tuned:
```
lr = 0.0005
batch_size = 32
epoch = 10
patience = 10
weight_decay = 0.0005
embed_dim = 64
n_transformer_layer = 2
n_attn_head = 4
max_triplet_len = 1000
n_feat = 51
demo_dim = 0
dropout = 0.3
n_cluster = 10
seed = 3
```