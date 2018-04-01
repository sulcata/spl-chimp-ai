import sys
import re
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier, AdaBoostClassifier, \
                             BaggingClassifier, VotingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, \
                                    StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn_pandas import DataFrameMapper, gen_features, cross_val_score


tier_list = ['rbyou', 'gscou', 'advou', 'dppou', 'bwou', 'orasou',
             'smdou', 'smnu', 'smru', 'smuu', 'smou']

semis_matchups = [
    ('abr', 'cdumas', 'smou', 'w'),
    ('trosko', 'kory2600', 'smou', 'w'),
    ('poketcggamer1288', 'bushtush', 'smuu', 'w'),
    ('soulgazer', 'chillshadow', 'smru', 'l'),
    ('rodriblutar', 'lax', 'smnu', 'l'),
    ('marilli', 'croven', 'smdou', 'l'),
    ('updatedkanto', 'hiye', 'orasou', 'l'),
    ('soulwind', 'zorodark', 'bwou', 'w'),
    ('void', 'wethreekings', 'dppou', 'w'),
    ('astamatitos', 'undisputed', 'advou', 'l'),
    ('fear', 'tiba', 'gscou', 'w'),
    ('theidiotninja', 'roudolf13', 'rbyou', 'l'),
    ('z0mog', '-tsunami-', 'smou', 'w'),
    ('psychicmewtwo', 'gondra', 'smou', 'w'),
    ('christo', 'pak', 'smuu', 'w'),
    ('silentverse', 'r!cardo', 'smru', 'l'),
    ('cbjosealtuve', 'zukushiku', 'smnu', 'w'),
    ('stax', 'kaori', 'smdou', 'l'),
    ('blunder', 'tdk', 'orasou', 'l'),
    ('dice', 'bkc', 'bwou', 'w'),
    ('bluewind', 'roscoe', 'dppou', 'l'),
    ('ud', 'jirachee', 'advou', 'l'),
    ('lavos', 'choolio', 'gscou', 'w'),
    ('earthworm', 'metalgro$$', 'rbyou', 'l')
]

finals_matchups = [
    ('abr', 'tdk', 'smou', 'N/A'),
    ('trosko', 'gondra', 'smou', 'N/A'),
    ('poketcggamer1288', 'pak', 'smuu', 'N/A'),
    ('soulgazer', '-tsunami-', 'smru', 'N/A'),
    ('rodriblutar', 'zukushiku', 'smnu', 'N/A'),
    ('marilli', 'kaori', 'smdou', 'N/A'),
    ('updatedkanto', 'r!cardo', 'orasou', 'N/A'),
    ('soulwind', 'bkc', 'bwou', 'N/A'),
    ('void', 'roscoe', 'dppou', 'N/A'),
    ('astamatitos', 'jirachee', 'advou', 'N/A'),
    ('fear', 'choolio', 'gscou', 'N/A'),
    ('theidiotninja', 'metalgro$$', 'rbyou', 'N/A')
]


def append_player_subscripts(l):
    return [*(s+'_x' for s in l), *(s+'_y' for s in l)]


def to_tiers(value):
    # split the list of tiers on forward slashes
    if not isinstance(value, str):
        return set()
    return set(re.sub(r"\s", "", value).lower().split("/"))


def to_record_list(value):
    # split a record of the form 'W-L-T'
    wins, losses, ties = re.sub(r"\s", "", value).split("-")
    return int(wins), int(losses), int(ties)


def to_week_data(value):
    # return W/L/T, opponent, tier
    if value == 'nan':
        return 'N/A', 'N/A', 'N/A'
    match = re.search(r"^(w|l) vs (.+?) \((.+?)\)$", value)
    result = match.group(1)
    opponent = re.sub(r"\s", "", match.group(2))
    tier = re.sub(r"\s", "", match.group(3))
    return result, opponent, tier


def to_week_label(n):
    if n == 11: return 'final'
    if n == 10: return 'semis'
    return 'week{}'.format(n)


def weeks(begin=1, end=11):
    return [to_week_label(i) for i in range(begin, end+1)]


def retrieve_spl9_overall_records():
    data = pd.read_csv("./data/spl-9-overall-records.csv")
    data.rename(columns=lambda col: re.sub(r"\W", "", col.lower()),
                inplace=True)

    del data['sorter']

    data = data.apply(lambda s: s.astype(str).str.lower())

    data['cost'] = pd.to_numeric(data['cost'])

    data['player'] = data['player'].str.replace(r"\s", "")

    for week in weeks():
        week_data = data[week].map(to_week_data)
        del data[week]
        data[week+'-result'] = week_data.map(lambda l: l[0])
        data[week+'-opponent'] = week_data.map(lambda l: l[1])
        data[week+'-tier'] = week_data.map(lambda l: l[2])

    record_lists = data['record'].map(to_record_list)
    del data['record']
    data['wins'] = record_lists.map(lambda l: l[0])
    data['losses'] = record_lists.map(lambda l: l[1])
    data['ties'] = record_lists.map(lambda l: l[2])

    tiers = data['tiers'].map(to_tiers)
    del data['tiers']
    for tier in tier_list:
        data[tier] = tiers.map(lambda tiers: tier in tiers)

    data.set_index('player', drop=False, inplace=True)

    return data


def get_matchups(records):
    def create_matchup(player1, player2):
        if player1 == 'N/A' or player2 == 'N/A':
            return None
        return player1, player2
    matchups = pd.DataFrame()
    players = records['player']
    for week in weeks():
        opponents = records[week+'-opponent']
        week_matchups = pd.Series(players.combine(opponents, create_matchup) \
                                         .dropna() \
                                         .get_values())
        matchups[week] = week_matchups
    return matchups


def get_matchups_info(records, matchups, weeks):
    info = pd.DataFrame()
    for week in weeks:
        week_info = pd.DataFrame()
        week_info[['player-x', 'player-y']] = matchups[week].dropna().apply(pd.Series)
        week_info['week'] = week
        week_info = week_info.merge(records, how='left',
                                    left_on='player-x', right_on='player')
        week_info = week_info.merge(records, how='left',
                                    left_on='player-y', right_on='player')
        info = info.append(week_info)
    info.set_index(['week', 'player-x', 'player-y'], inplace=True)
    return info


def get_instances(info, end_week, window_size):
    instances = pd.DataFrame()
    copy_columns = append_player_subscripts(['cost', *tier_list])
    for w in range(window_size+1, end_week+1):
        week = to_week_label(w)
        week_instances = pd.DataFrame()
        week_info = info.loc[week]
        week_instances[copy_columns] = week_info[copy_columns]
        week_instances['result'] = week_info[week+'-result_x']
        week_instances['tier'] = week_info[week+'-tier_x']
        week_instances['player-x'] = week_info['player_x']
        week_instances['player-y'] = week_info['player_y']
        week_instances['week'] = week
        begin = w-window_size
        end = w-1
        for p in 'x', 'y':
            for lookback_w, lookback_week in enumerate(weeks(begin, end), 1):
                week_instances['result-{}_{}'.format(lookback_w, p)] = week_info['{}-result_{}'.format(lookback_week, p)]
                week_instances['tier-{}_{}'.format(lookback_w, p)] = week_info['{}-tier_{}'.format(lookback_week, p)]
        instances = instances.append(week_instances)
    instances.set_index(['week', 'player-x', 'player-y'], inplace=True)
    return instances


def add_week(records, target_week, target_matchups):
    week = to_week_label(target_week)
    records = records.copy()
    for player_x, player_y, tier, result in target_matchups:
        records.loc[player_x, week+'-opponent'] = player_y
        records.loc[player_x, week+'-tier'] = tier
        records.loc[player_x, week+'-result'] = result
        records.loc[player_y, week+'-opponent'] = player_x
        records.loc[player_y, week+'-tier'] = tier
        records.loc[player_y, week+'-result'] = 'w' if result == 'l' else 'l'
    return records

def main():
    # set parameters and retrieve data
    begin_week = 1
    end_week = 11
    window_size = 5

    records = retrieve_spl9_overall_records()
    records = add_week(records, 10, semis_matchups)
    records = add_week(records, end_week, finals_matchups)
    matchups = get_matchups(records)
    info = get_matchups_info(records, matchups, weeks(begin_week, end_week))
    instances = get_instances(info, end_week, window_size)
    train_instances = instances.loc[[*weeks(begin_week, end_week-1)]]
    predict_instances = instances.loc[to_week_label(end_week)]

    # Create label transformations and attribute normalizations
    attribute_mapper = DataFrameMapper([
        *gen_features(columns=[['cost_x'], ['cost_y']],
                      classes=[{'class': StandardScaler}]),
        *gen_features(
            columns=append_player_subscripts([
                *('result-{}'.format(w+1) for w in range(window_size)),
                *('tier-{}'.format(w+1) for w in range(window_size))
            ]),
            classes=[LabelBinarizer]
        ),
        *gen_features(columns=append_player_subscripts(tier_list),
                      classes=[LabelBinarizer]),
        ('tier', LabelBinarizer())
    ])
    label_mapper = DataFrameMapper([
        ('result', LabelBinarizer())
    ])

    X = attribute_mapper.fit_transform(train_instances.copy())
    y = label_mapper.fit_transform(train_instances.copy()).ravel()
    X_act = attribute_mapper.transform(predict_instances.copy())
    y_act = label_mapper.transform(predict_instances.copy()).ravel()

    print(X.shape)
    print(X_act.shape)

    seed = 2718281828
    validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    with_prob = 'accuracy'
    without_prob = 'accuracy'

    trees_clf = RandomizedSearchCV(ExtraTreesClassifier(random_state=seed),
                                   cv=validation, n_iter=500,
                                   random_state=seed, scoring=with_prob,
                                   param_distributions={
        'n_estimators': stats.randint(low=40, high=200),
        'max_features': stats.uniform(loc=0.01, scale=0.99),
        'max_depth': stats.randint(low=1, high=10)
    })
    trees_clf.fit(X, y)
    print("Extra Trees:")
    print(trees_clf.best_params_)
    print(trees_clf.best_score_)
    print(accuracy_score(y_act, trees_clf.predict(X_act)))

    grad_clf = RandomizedSearchCV(GradientBoostingClassifier(random_state=seed),
                                  cv=validation, n_iter=500,
                                  random_state=seed, scoring=with_prob,
                                  param_distributions={
        'loss': ['exponential'],
        'learning_rate': stats.uniform(loc=1, scale=3),
        'n_estimators': stats.randint(low=40, high=200),
        'max_depth': stats.randint(low=1, high=10),
        'min_samples_split': stats.randint(low=2, high=13),
        'max_features': stats.uniform(loc=0.01, scale=0.99)
    })
    grad_clf.fit(X, y)
    print("Gradient Boosting:")
    print(grad_clf.best_params_)
    print(grad_clf.best_score_)
    print(accuracy_score(y_act, grad_clf.predict(X_act)))

    forest_clf = RandomizedSearchCV(RandomForestClassifier(random_state=seed),
                                    cv=validation, n_iter=500,
                                    random_state=seed, scoring=with_prob,
                                    param_distributions={
        'n_estimators': stats.randint(low=10, high=100),
        'max_features': stats.uniform(loc=0.01, scale=0.99),
        'max_depth': stats.randint(low=1, high=10)
    })
    forest_clf.fit(X, y)
    print("Random Forest:")
    print(forest_clf.best_params_)
    print(forest_clf.best_score_)
    print(accuracy_score(y_act, forest_clf.predict(X_act)))

    ada_clf = RandomizedSearchCV(AdaBoostClassifier(DecisionTreeClassifier(),
                                                    random_state=seed),
                                 cv=validation, n_iter=500,
                                 random_state=seed, scoring=with_prob,
                                 param_distributions={
        'base_estimator__max_depth': stats.randint(low=1, high=10),
        'n_estimators': stats.randint(low=10, high=100),
        'learning_rate': stats.uniform(loc=1.29, scale=0.06)
    })
    ada_clf.fit(X, y)
    print("AdaBoost:")
    print(ada_clf.best_params_)
    print(ada_clf.best_score_)
    print(accuracy_score(y_act, ada_clf.predict(X_act)))

    bagging_clf = RandomizedSearchCV(BaggingClassifier(random_state=seed),
                                     cv=validation, n_iter=500,
                                     random_state=seed, scoring=with_prob,
                                     param_distributions={
        'n_estimators': stats.randint(low=57, high=75),
        'max_samples': stats.randint(low=6, high=9)
    })
    bagging_clf.fit(X, y)
    print("Bagging:")
    print(bagging_clf.best_params_)
    print(bagging_clf.best_score_)
    print(accuracy_score(y_act, bagging_clf.predict(X_act)))

    svc_clf = RandomizedSearchCV(SVC(random_state=seed),
                                 cv=validation, n_iter=500,
                                 random_state=seed, scoring=with_prob,
                                 param_distributions={
        'kernel': ['poly'],
        'degree': stats.randint(low=2, high=4),
        'C': stats.uniform(loc=1, scale=13),
        'coef0': stats.uniform(loc=-7, scale=8)
    })
    svc_clf.fit(X, y)
    print("SVC:")
    print(svc_clf.best_params_)
    print(svc_clf.best_score_)
    print(accuracy_score(y_act, svc_clf.predict(X_act)))

    nn_clf = RandomizedSearchCV(MLPClassifier(random_state=seed),
                                cv=validation, n_iter=500,
                                random_state=seed, scoring=with_prob,
                                param_distributions={
        'activation': ['relu'],
        'solver': ['lbfgs'],
        'hidden_layer_sizes': [(7,)],
        'alpha': stats.uniform(loc=0.1e-4, scale=1.5e-4)
    })
    nn_clf.fit(X, y)
    print("Neural Network:")
    print(nn_clf.best_params_)
    print(nn_clf.best_score_)
    print(accuracy_score(y_act, nn_clf.predict(X_act)))

    base_voting_clf = VotingClassifier([
        ('ada', AdaBoostClassifier(DecisionTreeClassifier(random_state=seed))),
        ('svc', SVC(random_state=seed)),
        ('grad', GradientBoostingClassifier(random_state=seed))
    ])
    voting_clf = RandomizedSearchCV(base_voting_clf,
                                    cv=validation, n_iter=500,
                                    scoring=with_prob, param_distributions={
        'voting': ['hard', 'soft'],
        'svc__probability': [True],
        'svc__kernel': ['sigmoid'],
        'svc__gamma': stats.uniform(loc=5.1e-2, scale=0.1e-2),
        'svc__coef0': stats.uniform(loc=8.5e-2, scale=0.2e-2),
        'svc__C': stats.uniform(loc=5.4, scale=0.02),
        'ada__base_estimator__max_depth': [1],
        'ada__n_estimators': stats.randint(low=38, high=40),
        'ada__learning_rate': stats.uniform(loc=1.29, scale=0.06)
    })
    voting_clf.fit(X, y)
    print("Voting:")
    print(voting_clf.best_params_)
    print(voting_clf.best_score_)
    print(accuracy_score(y_act, voting_clf.predict(X_act)))


if __name__ == '__main__':
    main()
