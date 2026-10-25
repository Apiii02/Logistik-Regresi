
import sys
import numpy as np
import pandas as pd

CSV = r"e:\sem 5\pembelajaran mesin\bank.csv"
np.set_printoptions(precision=5, suppress=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ----------------------------------------------------------------------------
# Fungsi inti (sesuai materi)
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def cross_entropy(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_logreg(X, y, w, b, eta, epochs, verbose=False, names=None):
    """Batch gradient descent. Jika verbose=True tampilkan tabel per langkah."""
    N = len(y)
    history = []
    for ep in range(1, epochs + 1):
        z = X @ w + b                       # langkah 2
        p = sigmoid(z)                      # langkah 3
        L = cross_entropy(y, p)             # langkah 4
        J = L.mean()
        err = p - y                         # (sigma - y)
        dw = (err[:, None] * X).mean(axis=0)  # langkah 5 (gradien)
        db = err.mean()
        w_new = w - eta * dw
        b_new = b - eta * db
        history.append(J)
        if verbose:
            tab = pd.DataFrame(X, columns=names)
            tab["z"], tab["sigma"], tab["y"], tab["L"] = z, p, y, L
            tab["(sigma-y)"] = err
            for k, n in enumerate(names):
                tab[f"(s-y)*{n}"] = err * X[:, k]
            print(f"\n=== EPOCH {ep} ===  w={w}, b={b:.5f}, eta={eta}")
            print(tab.round(5).to_string(index=False))
            print(f"Total L = {L.sum():.5f}   J = {J:.5f}")
            print(f"dw = {dw}   db = {db:.5f}")
            print(f"w baru = {w_new}   b baru = {b_new:.5f}")
        w, b = w_new, b_new
    return w, b, history


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def metrics(y, yhat):
    TP = int(((y == 1) & (yhat == 1)).sum())
    TN = int(((y == 0) & (yhat == 0)).sum())
    FP = int(((y == 0) & (yhat == 1)).sum())
    FN = int(((y == 1) & (yhat == 0)).sum())
    acc = (TP + TN) / len(y)
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, akurasi=acc, presisi=prec, recall=rec, f1=f1)


# ----------------------------------------------------------------------------
# BAGIAN A : 5 data latih + 1 data uji
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV)
df["y"] = (df["deposit"] == "yes").astype(int)   # yes=1, no=0 (2 kelas)

FITUR = ["duration", "campaign"]
rng = np.random.RandomState(42)
yes_idx = rng.choice(df.index[df.y == 1], 3, replace=False)   # 3 kelas 1
no_idx = rng.choice(df.index[df.y == 0], 2, replace=False)    # 2 kelas 0
latih_idx = np.concatenate([yes_idx, no_idx])
sisa = df.index.difference(latih_idx)
uji_idx = rng.choice(sisa, 1, replace=False)

latih = df.loc[latih_idx, FITUR + ["deposit", "y"]].reset_index(drop=True)
uji = df.loc[uji_idx, FITUR + ["deposit", "y"]].reset_index(drop=True)

print("=" * 70)
print("LANGKAH 1 - 5 DATA LATIH (data asli)")
print("=" * 70)
print(latih.to_string())
print("\nData uji (1 data):")
print(uji.to_string())

# Praproses: standardisasi z-score memakai mean & std DATA LATIH saja
mu = latih[FITUR].mean().values
sd = latih[FITUR].std(ddof=0).values
Xl = (latih[FITUR].values - mu) / sd
yl = latih["y"].values.astype(float)
Xu = (uji[FITUR].values - mu) / sd

print("\nPraproses z-score: mean =", mu, " std =", sd)
print("Data latih terstandarisasi:\n", Xl)
print("Data uji terstandarisasi:\n", Xu)

# Parameter awal (sama dengan contoh di materi)
w0 = np.array([0.9, -0.1])
b0 = 0.1
eta = 0.1

print("\n" + "=" * 70)
print("LANGKAH 2 - PROSES LATIH (PSEDO) 2 EPOCH")
print("=" * 70)
w2, b2, hist = train_logreg(Xl, yl, w0.copy(), b0, eta, epochs=2,
                            verbose=True, names=FITUR)

print("\n" + "=" * 70)
print("LANGKAH 3 - KLASIFIKASI DATA UJI")
print("=" * 70)
z = float(Xu[0] @ w2 + b2)
p = float(sigmoid(z))
kelas = int(p >= 0.5)
print(f"x uji (std) = {Xu[0]}")
print(f"z = w1*x1 + w2*x2 + b = {z:.5f}")
print(f"P(y=1|x) = sigma(z) = {p:.5f}  ->  kelas prediksi = {kelas} "
      f"({'yes' if kelas else 'no'})")
print(f"kelas sebenarnya = {uji['y'][0]} ({uji['deposit'][0]})")

# Bagian A (4a) : latih lebih lama sampai konvergen, lalu uji
wA, bA, histA = train_logreg(Xl, yl, w0.copy(), b0, eta, epochs=1000)
pA = float(predict_proba(Xu, wA, bA)[0])
print(f"\n[Tambahan] Setelah 1000 epoch: w={wA}, b={bA:.5f}, J={histA[-1]:.5f}")
print(f"           P(y=1|x uji) = {pA:.5f} -> kelas {int(pA >= 0.5)}")

# ----------------------------------------------------------------------------
# BAGIAN B : seluruh data 80% : 20%
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BAGIAN B - SELURUH DATA, SPLIT 80% : 20%")
print("=" * 70)

X_all = df.drop(columns=["deposit", "y"])
X_all = pd.get_dummies(X_all, drop_first=True).astype(float)   # one-hot kategorik
y_all = df["y"].values.astype(float)

# split acak 80:20 (stratified manual agar proporsi kelas terjaga)
rng = np.random.RandomState(42)
tr, te = [], []
for c in (0, 1):
    idx = rng.permutation(np.where(y_all == c)[0])
    n_tr = int(round(0.8 * len(idx)))
    tr += list(idx[:n_tr])
    te += list(idx[n_tr:])
tr, te = np.array(tr), np.array(te)

mu_b = X_all.values[tr].mean(axis=0)
sd_b = X_all.values[tr].std(axis=0)
sd_b[sd_b == 0] = 1.0
Xtr = (X_all.values[tr] - mu_b) / sd_b
Xte = (X_all.values[te] - mu_b) / sd_b
ytr, yte = y_all[tr], y_all[te]
print(f"Jumlah data latih = {len(tr)}, data uji = {len(te)}, jumlah fitur = {Xtr.shape[1]}")

wB, bB, histB = train_logreg(Xtr, ytr, np.zeros(Xtr.shape[1]), 0.0,
                             eta=0.1, epochs=1000)
print(f"Loss J epoch 1 = {histB[0]:.5f}, epoch 100 = {histB[99]:.5f}, "
      f"epoch 1000 = {histB[-1]:.5f}")

for nama, X_, y_ in (("LATIH", Xtr, ytr), ("UJI", Xte, yte)):
    yhat = (predict_proba(X_, wB, bB) >= 0.5).astype(int)
    m = metrics(y_, yhat)
    print(f"\nEvaluasi data {nama}:")
    print(f"  Confusion matrix  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
    print(f"  Akurasi={m['akurasi']:.4f}  Presisi={m['presisi']:.4f}  "
          f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}")

# Pembanding: scikit-learn (opsional)
try:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    print(f"\n[Pembanding sklearn] akurasi uji = {clf.score(Xte, yte):.4f}")
except ImportError:
    pass

# Fitur paling berpengaruh (koefisien terbesar, data sudah distandarisasi)
koef = pd.Series(wB, index=X_all.columns).sort_values(key=np.abs, ascending=False)
print("\n10 koefisien terbesar (|w|):")
print(koef.head(10).round(4).to_string())