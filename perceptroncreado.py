import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from perceptronprofe import Perceptron
from mlxtend.classifier import Perceptron




X, y = iris_data()
X = X [:, [0,2]]
X = X[0:100]
y = y[0:100]
#y = np.where(y == 0, -1, 1)


#preprocesamiento, estandsariza la info
#print (X[:,0].mean())
X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,0].mean()) / X[:,1].std()


ppn = Perceptron(epochs = 5, eta = 0.05, random_seed = 0, print_progress = 3)
ppn.fit(X,y)

'''ppn = Perceptron(epochs = 5|0, eta = 0.1)
ppn.train(X,y)'''


plot_decision_regions(X, y, clf = ppn)
plt.title('perceptron - regla de rosenblantt')
plt.show()

print ('Pesos: %s' % ppn.w_)