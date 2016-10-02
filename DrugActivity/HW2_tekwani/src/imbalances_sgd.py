from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, roc_auc_score
from feature_creation import X_train, y_train
from feature_creation import selector, idx, df_reduced_train
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss, CondensedNearestNeighbour, RandomUnderSampler
from imblearn.under_sampling import OneSidedSelection, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.linear_model import SGDClassifier

imbalances = [RandomUnderSampler(),
              TomekLinks(),
              ClusterCentroids(),
              NearMiss(version=1),
              NearMiss(version=2),
              NearMiss(version=3),
              CondensedNearestNeighbour(size_ngh=3, n_seeds_S=51),
              OneSidedSelection(size_ngh=51, n_seeds_S=51),
              InstanceHardnessThreshold(),
              RandomOverSampler(ratio='auto'),
              SMOTE(ratio='auto', kind='regular'),
              SMOTE(ratio='auto', kind='borderline1'),
              SMOTE(ratio='auto', kind='borderline2'),
              SMOTETomek(ratio='auto'),
              SMOTEENN(ratio='auto')]


clf = SGDClassifier(n_iter=10000, loss='modified_huber', penalty='elasticnet', alpha=0.07,
                    shuffle=True)

best_dict = {}


def sampling():
    for imbalance in imbalances:
        X_tr, X_te , y_tr, y_te = train_test_split(df_reduced_train.values, y_train, test_size=0.4, stratify=y_train)

        X_tr, y_tr = imbalance.fit_sample(X_tr, y_tr)

        print ("------------")
        print ("%s " % imbalance)
        print ("------------")

        clf.fit(X_tr, y_tr)
        print ("-------------")

        print("%s   " % clf)
        print('-----------------')
        print("")

        print("Accuracy score", accuracy_score(y_te, clf.predict(X_te)))
        print('auc', roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        print("")

        print(classification_report(y_te, clf.predict(X_te)))
        print("")

        print('-----------------')
        best_dict[imbalance] = [clf, roc_auc_score(y_te, clf.predict(X_te))]


sampling()

print best_dict