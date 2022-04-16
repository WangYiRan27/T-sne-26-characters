from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat

vision=loadmat('C:\\Users\\PCF\\Desktop\\IDVRM\\IDVRM\\data\\az\\matlab.mat')
Y_train = vision['EEGTra']
Y_test = vision['EEGTes']
X_train = vision['stimTra']
X_test = vision['stimTes']
LABEL = vision['LABEL']

digits = load_digits()
print(digits.data.shape)
print(LABEL.shape)
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(Y_train)
X_pca = PCA(n_components=2).fit_transform(Y_train)

ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=LABEL, label="t-SNE")
plt.legend()
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=LABEL, label="PCA")
plt.legend()
plt.savefig('images/digits_tsne-pca.png', dpi=120)
plt.show()