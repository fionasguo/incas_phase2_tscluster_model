import logging
import numpy as np
import pandas as pd
from collections import Counter
import re
import pickle


def preprocess(ids,tweets,users,timestamps,args,user_tot_tweet_threshold=5,agg_time_period='D'):
    df = pd.DataFrame([ids,tweets,users,timestamps])
    df = df.T
    df.columns = ['id','contentText','twitterAuthorScreenname','timePublished']
    df['timePublished'] = pd.to_datetime(df['timePublished'],unit='ms')
    # only take users with more than some tweets
    active_users = df.groupby(['twitterAuthorScreenname'])['id'].count()
    tot_user_num = len(active_users)
    active_users = active_users[active_users>user_tot_tweet_threshold]
    active_user_set = set(active_users.index)
    df = df[df['twitterAuthorScreenname'].isin(active_user_set)]
    df = df.reset_index(drop=True)
    if len(df) < 500:
        logging.info('too little data, cannot train')
        return None,None
    logging.info(f'data with more than {user_tot_tweet_threshold} tweets - shape: {df.shape}, number of these active users: {len(active_user_set)}, frac of active users: {len(active_users)/tot_user_num}')
    logging.info(f'stats of num tweets from active users: mean={active_users.mean()}, std={active_users.std()}')
    
    logging.info('start building hashtag features')
    # find hashtags for each user
    hashtags = [re.findall(r'#\S+', text) for text in tweets]
    logging.info(f"done detecting hashtags for individual tweets, avg # hashtags for individual user: {np.mean([len(h) for h in hashtags])}")
    logging.info("getting top 50 hashtags in all text")
    # find top 50 hashtags in all text
    all_hashtags = [h for user_hashtags in hashtags for h in user_hashtags]
    logging.info(f'number of all hashtags found in the data: {len(all_hashtags)}')
    all_hashtags = [h[0] for h in Counter(all_hashtags).most_common(50)]
    all_hashtags_set = set(all_hashtags)
    logging.info(f"top hashtags extracted")

    onehot_hashtags = [[1 if h in user_hashtags else 0 for h in all_hashtags] for user_hashtags in hashtags]
    df[all_hashtags] = onehot_hashtags
    logging.info(f'one hot encoding done, top hashtags and their freq:\n{df[all_hashtags].mean(axis=0)}')
    # aggregate to user ts data
    user_ts_data = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])[all_hashtags].sum()
    user_ts_data['twitter_count'] = df.groupby(['twitterAuthorScreenname',pd.Grouper(freq=agg_time_period,key='timePublished')])['id'].count()
    logging.info(f'raw user hashtag ts data - shape: {user_ts_data.shape}')
    # fill the time series with the entire time range
    entire_time_range = pd.date_range(start=df['timePublished'].min(),end=df['timePublished'].max(),freq=agg_time_period)
    user_ts_data = user_ts_data.reindex(pd.MultiIndex.from_product([user_ts_data.index.levels[0],entire_time_range],names=['twitterAuthorScreenname','timePublished']),fill_value=0)
    logging.info(f'user ts data filled up to entire time range - shape: {user_ts_data.shape}; number of users: {len(active_user_set)}, len of entire time range: {len(entire_time_range)}')

    # transform into 3-d np array
    ts_array = np.array(user_ts_data.groupby(level=0).apply(lambda x: x.values.tolist()).tolist())
    logging.info(f'shape of np array for the ts data: {ts_array.shape}')
    pickle.dump(ts_array, open(args['output_dir']+'/hashtag_ts_data.pkl','wb'))
    logging.info('finished saving hashtag ts data')
    
    return_user_lst = list(user_ts_data.groupby(level=0)['twitter_count'].first().index)

    return args['output_dir']+'/hashtag_ts_data.pkl', return_user_lst