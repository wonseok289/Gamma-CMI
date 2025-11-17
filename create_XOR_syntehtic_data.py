import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def create_XOR(n, noisy_x, delta, seed=42):  # 전체 데이터 개수, noisy_x의 개수, delta noise
    '''
    기본 feature 구성: x_1, x_2, noisy x_s
    y = x_1 ^ x_2로 구성 (XOR)
    delta값에 따라 다른 noisy_x가 샘플링될 때 y와 동일한 값의 샘플이 더 많이 생성됨 (0 ~ 0.5) 
    '''
    assert 0.0 <= delta <= 0.5

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x1 = np.random.binomial(1, 0.5, size=n)
    x2 = np.random.binomial(1, 0.5, size=n)
    y = (x1 ^ x2).astype(int)

    p = 0.5 + delta * (2*y - 1)
    p = p.reshape(-1, 1)
    noisy = np.random.binomial(1, p, size=(n, noisy_x))

    X = np.concatenate(
        [x1.reshape(-1, 1), x2.reshape(-1, 1), noisy],
        axis=-1
    )

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
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    X_train = torch.from_numpy(X_train).long()
    X_val   = torch.from_numpy(X_val).long()
    X_test  = torch.from_numpy(X_test).long()

    y_train = torch.from_numpy(y_train).long()
    y_val   = torch.from_numpy(y_val).long()
    y_test  = torch.from_numpy(y_test).long()

    torch.save(X_train, "X_train.pt")
    torch.save(X_val,   "X_val.pt")
    torch.save(X_test,  "X_test.pt")

    torch.save(y_train, "y_train.pt")
    torch.save(y_val,   "y_val.pt")
    torch.save(y_test,  "y_test.pt")

    return None

if __name__ == "__main__":
    create_XOR(
        n=10000,    # 샘플 수
        noisy_x=6,  # noisy feature 개수
        delta=0.5,  # noisy feature 편향 strength
        seed=42
    )
    print("생성 완료")
