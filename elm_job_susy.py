__author__ = 'ozgurcatak'

from mrjob.job import MRJob
import random, numpy as np, sys, os, uuid, errno, time
from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib

M = 2
nh = 5
T = 5

srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

# clf = GenELMClassifier(hidden_layer=srhl_tanh)
clf = GenELMClassifier(hidden_layer=srhl_rbf)
# clf = GenELMClassifier(hidden_layer=srhl_tribas)
# clf = GenELMClassifier(hidden_layer=srhl_hardlim)



class ELMTraining(MRJob):

    def mapper(self, _, line):
        k = random.randint(1,M)
        yield k, (line)

    def reducer(self, key, values):
        D = np.zeros((1, 1))

        f_tmp = open("tmp_val_" + str(key) + ".txt", "w");
        for v in values:
            f_tmp.write(v + "\n")
        f_tmp.close()

        D = np.loadtxt("tmp_val_" + str(key) + ".txt", delimiter=",")

        bdt_discrete = AdaBoostClassifier(
            clf,
            n_estimators=T,
            learning_rate=5,
            algorithm= "SAMME")
        m,n = D.shape

        X = D[:, 0:n-1]
        y = D[:, n-1]
        D = None

        t = time.time()
        bdt_discrete.fit(X, y)
        elapsed = time.time() - t

        X = None
        y = None
        
		
        test_file = ds_name.replace("/ds/","/test/")
        test_file = test_file.replace("\\ds\\","\\test\\")
        print '*'*50,test_file
		
        D = np.loadtxt(test_file, delimiter=",")
        X_test = D[:, 0:n-1]
        y_test = D[:, n-1]

        y_hat = bdt_discrete.predict(X_test)
        X_test = None
        cr = classification_report(y_test, y_hat)
        acc = accuracy_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat)
        recall = recall_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)

        f_tmp = str(uuid.uuid1())
        # joblib.dump(bdt_discrete, model_folder + "/" + f_tmp)

        # print cr
        # print acc
        # print '*'*50

        f_res = open(res_file,"a")

        f_res.write(str(T) + "\t" + str(M) + "\t" + str(key) + "\t" + str(nh) + "\t" + f_tmp + "\t"
                    + str(acc) + "\t" + str(prec) + "\t" + str(recall) + "\t"
                    + str(f1) + "\t" + str(elapsed) + "\n")

        f_res.close()


if __name__ == '__main__':
    ds_name = os.path.abspath(sys.argv[1])
    res_file = os.path.abspath(sys.argv[1].replace(".txt", "").replace("ds/", "") + "-results.txt")

    model_folder = "models/" + sys.argv[1].replace(".txt", "").replace("ds/", "")
    try:
        os.makedirs(model_folder)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(model_folder):
            pass
        else:
            raise
    model_folder = os.path.abspath(model_folder)

    print ds_name
    print model_folder
    for ind in range(10):	
		for T in range (50,51,5):
			for M in range(50, 30,-1):
				for nh in range(25, 26, 25):
					ELMTraining.run()