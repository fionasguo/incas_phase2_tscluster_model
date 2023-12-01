import os
import time
import json
import numpy as np
import logging

from data_processing import preprocess
import TSCluster


os.environ["TOKENIZERS_PARALLELISM"] = "false"

######################### Parameters #########################
USER_TOT_TWEET_THRESHOLD = 1
AGG_TIME_PERIOD = '12H'

######################### Annotate #########################


def format_output(ids, preds):
    """
    Format predictions into Annotation instances.

    Args:
        ids: list of message ids, len=n_messages
        preds_conf: list, shape=(n_messages,)

    Returns:
        A dict whose keys are message IDs and values are lists of annotations.
    """
    annotations = {}
    for id, p in zip(ids, preds):
        if id in annotations: continue

        annotations[id] = [
            Annotation(
                confidence=1,
                attributeName="contentText",
                providerName="ta1-usc-isi",
                text="synchronized user group",
                type=str(p),
            )
        ]


    return annotations


def train(data, args):
    start_time = time.time()
    logging.info('Start training...')

    # set up trainer
    trainer = TSCluster.Trainer(data, args)

    trainer.train()

    logging.info(
        f"Finished training data. Time: {time.time()-start_time}"
    )

    return trainer


def test(data,args,trainer=None):
    start_time = time.time()

    logging.info('============ Evaluation on Test Data ============= \n')
    # evaluate with the best model just got from training or with the model given from config
    if trainer is not None:
        eval_model_path = args['output_dir'] + '/model_weights.h5'
    elif args.get('trained_model_dir') is not None:
        eval_model_path = args['trained_model_dir']
    else:
        raise ValueError('Please provide a model for evaluation.')

    # TODO: implement evaluation with gt
    preds = TSCluster.evaluate(data, eval_model_path, args)

    # # output predictions
    # with open(args['output_dir'] + '/mf_preds.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(mf_preds)

    logging.info(
        f"Finished evaluating test data. Time: {time.time()-start_time}"
    )
    return preds


def annotate(data):
    """
    Args:
        tweets: list of dictionaries [{'id':xxx,'contentText':xxx,...},...]
    Returns:
        A dict whose keys are message IDs and values are lists of annotations.
    """
    ## process data
    start_time = time.time()
    logging.debug('Start processing data...')

    if not data or len(data) == 0: return {}
    if isinstance(data, dict): data = [data]

    ids = [i['id'] for i in data]
    tweets = [' ' if not i['contentText'] else i['contentText'] for i in data]
    # users = [np.nan if not json.loads(json.loads(i['mediaTypeAttributes'])['twitterData'])['twitterAuthorScreenname'] else json.loads(json.loads(i['mediaTypeAttributes'])['twitterData'])['twitterAuthorScreenname'] for i in data]
    users = [i['mediaTypeAttributes']['twitterData']['twitterAuthorScreenname'] if i.get('mediaTypeAttributes') and i['mediaTypeAttributes'].get('twitterData') and i['mediaTypeAttributes']['twitterData'].get('twitterAuthorScreenname') else np.nan for i in data]
    timestamps = [np.nan if not i['timePublished'] else int(i['timePublished']) for i in data]
    mask = [True if (u is not np.nan) and (t is not np.nan) else False for u,t in zip(users,timestamps)]
    # preprocessing tweets
    ts_data_dir,return_user_lst = preprocess(
        [x for x,m in zip(ids,mask) if m],
        [x for x,m in zip(tweets,mask) if m],
        [x for x,m in zip(users,mask) if m],
        [x for x,m in zip(timestamps,mask) if m],
        args,
        user_tot_tweet_threshold=USER_TOT_TWEET_THRESHOLD,
        agg_time_period=AGG_TIME_PERIOD
    )
    if ts_data_dir is None:
        # too little data, cannot train, assgin everyone to the same cluster 0
        all_preds = [0]*len(tweets)
    else:
        # train and inference
        datasets = TSCluster.read_data(
                    ts_data_dir=ts_data_dir,
                    demo_data_dir=args['demo_data_dir'],
                    gt_data_dir=args['gt_dir'],
                    max_triplet_len=args['max_triplet_len'],
                    data_split='no'
                )

        trainer = train(datasets, args)
        return_user_preds = test(datasets,args,trainer)
        logging.info(f'len of return_user_preds: {len(return_user_preds)}, len of return_user_lst: {len(return_user_lst)}')
        logging.info(return_user_preds)

        # assign the same user label to all their tweets
        all_preds = np.full((len(tweets),1),-1)
        users = np.array(users)
        for p,u in zip(return_user_preds,return_user_lst):
            print(p,u)
            print(users==u)
            print(all_preds[users==u,0])
            all_preds[users==u,0] = p
    
    with open(args['output_dir']+'/predictions.csv','w') as f:
        for p in all_preds:
            f.write(str(p)+'\n')

    # format to INCAS required output
    # outputs = format_output(ids, all_preds)
    # logging.debug(f'Finished evalutating. Time: {time.time()-start_time}.')

    return all_preds

if __name__ == "__main__":
    # logger
    # logging.disable(logging.WARNING)
    TSCluster.create_logger()
    # args
    root_dir = os.path.dirname(os.path.realpath(__file__))
    _, args = TSCluster.get_training_args(root_dir)
    # read file
    with open(args['ts_data_dir'], 'r') as f:
        data = [json.loads(x) for x in f.read().splitlines()]
    more_data = []
    for _ in range(30):
        more_data.extend(data)
    logging.info(f'raw data length: {len(more_data)}')
            
    annotate(more_data)

