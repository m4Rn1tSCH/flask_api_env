'''
Split Data
Splitting a given databse into labels and feautures, and train and test sets
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def split_data(df, test_size=0.2, label='amount_mean_lag7'):
    '''
    Parameters
    ----------
    df : dataframe to split into label, features and train, test sets
    test_size : num from 0 - 1, the size of test set relative to train set. Default is 0.2
    label : column on dataframe to use as label. Default is 'amount_mean_lag7'

    Returns
    -------
    [X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax, y_train, y_test]
    '''
    #drop target variable in feature df
    model_features = df.drop(labels=label, axis=1)
    model_label = df[label]

    if label == 'amount_mean_lag7':
        # To round of amount and lessen data complexity
        if model_label.dtype == 'float32':
            model_label = model_label.astype('int32')
        elif model_label.dtype == 'float64':
            model_label = model_label.astype('int64')
        else:
            print("model label has unsuitable data type!")

    # splitting data into train and test values
    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        random_state=7,
                                                        shuffle=True,
                                                        test_size=test_size)

    #create a validation set from the training set
    print(f"Shapes X_train:{X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    #STD SCALING
    #standard scaler works only with maximum 2 dimensions
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    #transform test data with the object learned from the training data
    X_test_scaled = scaler.transform(X_test)

    #MINMAX SCALING
    #works with Select K Best
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    #transform test data with the object learned from the training data
    X_test_minmax = min_max_scaler.transform(X_test)

    return [X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax, y_train, y_test]
