import numpy as np

class SimplePreprocessor(object):
    def __init__(self, is_categorical):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, np.where(is_categorical)[0]),
                ('num', numeric_transformer, np.where(~is_categorical)[0])])

    def fit(self, x):
        return self.preprocessor.fit(x)

    def fit_transform(self, x):
        return self.preprocessor.fit_transform(x)

    def transform(self, x):
        return self.preprocessor.transform(x)
    
    
class MetaDataset(object):
    def __init__(self):
        self.datasets = []
        self.datasets_orig = []
        self.primary_y = []
        self.drift_types = []
        self.drops = []
        self.metrics_id = []
        self.meta_features = []

    def append(self, dataset, dataset_orig, primary_y, drift_type, drop, metrics_id, meta_features):
        self.datasets.append(dataset)
        self.datasets_orig.append(dataset_orig)
        self.primary_y.append(primary_y)
        self.drift_types.append(drift_type)
        self.drops.append(drop)
        self.metrics_id.append(metrics_id)
        self.meta_features.append(meta_features)

    def arrayfy(self):
        self.datasets = np.array(self.datasets)
        self.datasets_orig = np.array(self.datasets_orig)
        self.primary_y = np.array(self.primary_y)
        self.drift_types = np.array(self.drift_types)
        self.drops = np.array(self.drops)
        self.metrics_id = np.array(self.metrics_id)
        self.meta_features = np.array(self.meta_features)

    def shuffle(self):
        shuffled_indices = np.random.permutation(len(self.drift_types))

        self.datasets = self.datasets[shuffled_indices]
        self.datasets_orig = self.datasets_orig[shuffled_indices]
        self.primary_y = self.primary_y[shuffled_indices]
        self.drops = self.drops[shuffled_indices]
        self.drift_types = self.drift_types[shuffled_indices]
        self.metrics_id = self.metrics_id[shuffled_indices]


class ReferenceTask(object):
    # drop is zero
    def __init__(self, model, X_src_orig, y_src, preprocess, is_categorical):
        self.model = model
        self.X_orig = X_src_orig # val set
        self.y = y_src # val set
        self.preprocess = preprocess
        self.is_categorical = is_categorical

        