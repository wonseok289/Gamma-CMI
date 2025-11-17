import numpy as np
import torch

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    d = 6   # label 후보 feature 개수
    n = 10000  # 총 데이터 개수

    X = np.random.binomial(n=1, p=0.5, size=(n, d))
    indicators = np.random.choice(a=d, size=(n, 1), replace=True)
    y = X[np.arange(n), indicators.flatten()]
    X = np.concatenate([X, indicators], axis=-1)

    # 8:1:1 분할 

    # train, (val+test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    # val, test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.10,
        random_state=42,
        shuffle=True,
    )

    X_train = torch.from_numpy(X_train).float()
    X_val   = torch.from_numpy(X_val).float()
    X_test  = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).float()
    y_val   = torch.from_numpy(y_val).float()
    y_test  = torch.from_numpy(y_test).float()

    torch.save(X_train, "X_train.pt")
    torch.save(X_val,   "X_val.pt")
    torch.save(X_test,  "X_test.pt")

    torch.save(y_train, "y_train.pt")
    torch.save(y_val,   "y_val.pt")
    torch.save(y_test,  "y_test.pt")