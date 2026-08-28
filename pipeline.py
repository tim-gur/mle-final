from scripts.process_data import process_data
from scripts.als_recommendations import als_recommendations
from scripts.candidates import candidates
from scripts.features import features
from scripts.fit import fit
from scripts.evaluate import evaluate

process_data()
print('data processed')

als_recommendations()
print('als recommendations done')

candidates_for_train, candidates_to_rank = candidates()
print('initial candidates done')

features(candidates_for_train, candidates_to_rank)
print('features merged with candidates')

fit()
print('model fit')

evaluate()
print('metrics done')