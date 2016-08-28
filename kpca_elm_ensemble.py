from __future__ import division
import sqlalchemy,csv
from array import array
import numpy as np
from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer
import random
import sklearn
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,confusion_matrix
from datetime import datetime
import operator

engine = sqlalchemy.create_engine('mysql://root@localhost') # connect to server
engine.execute("USE kddcup99") # select new db

def tic():
	#Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
	else:
		print "Toc: start time not set"
		
def entropy(X):
    probs = [np.mean(X == c) for c in set(X)]
    return np.sum(-p * np.log2(p) for p in probs)

def trainModel():
	cnt = 4898942
	bagsize = 10;
	nh = 20;
	pca_components = 10;
	range_count = int(cnt/bagsize)

	srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
	srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
	srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
	srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

	#clf = GenELMClassifier(hidden_layer=srhl_tanh);
	#clf = GenELMClassifier(hidden_layer=srhl_rbf);
	clf = GenELMClassifier(hidden_layer=srhl_tribas);
	#clf = GenELMClassifier(hidden_layer=srhl_hardlim);

	sql = "select case label WHEN 'back' then 0 WHEN 'land' then 0 WHEN 'neptune' then 0 WHEN 'pod' then 0 WHEN 'smurf' then 0 WHEN 'teardrop' then 0 WHEN 'ipsweep' then 1 WHEN 'nmap' then 1 WHEN 'portsweep' then 1 WHEN 'satan' then 1 WHEN 'ftp_write' then 2 WHEN 'guess_passwd' then 2 WHEN 'imap' then 2 WHEN 'multihop' then 2 WHEN 'phf' then 2 WHEN 'spy' then 2 WHEN 'warezclient' then 2 WHEN 'warezmaster' then 2 WHEN 'buffer_overflow' then 3 WHEN 'loadmodule' then 3 WHEN 'perl' then 3 WHEN 'rootkit' then 3 when 'normal' then 4  end as label1 ,duration,src_bytes,dst_bytes,wrong_fragment,urgent,hot,num_failed_logins,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,countx,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate from transaction1 where connection in $ids$ or label in( 'buffer_overflow','warezmaster','guess_passwd') order by rand()";

	ids = [ random.randrange(512,4898942) for i in range(range_count)]
	ids = str(ids).replace('[','(').replace(']',')')
	rec = engine.execute(sql.replace("$ids$",ids))
	b = open('test.csv', 'wb')
	w = csv.writer(b)
	for row in rec:
		w.writerow(row)
	b.close();
	D = np.loadtxt('tmp.csv', delimiter=',')
	X_test = D[:,1:38]
	y_test = D[:,0];
	D = None
	
	fout = open("results.txt","w");
		
	print bagsize
	for ind in range(bagsize):
		ids = [ random.randrange(512,4898942) for i in range(range_count)]
		ids = str(ids).replace('[','(').replace(']',')')
		tic()
		
		rec = engine.execute(sql.replace("$ids$",ids))
		b = open('tmp.csv', 'wb')
		w = csv.writer(b)
		for row in rec:
			w.writerow(row)
		b.close();
		toc()
		
		D = np.loadtxt('tmp.csv', delimiter=',')
		X = D[:,1:38]
		y = D[:,0];
		normalizer = sklearn.preprocessing.Normalizer().fit(X)
		X = normalizer.transform(X)
		
		ent = entropy(y)
		print "Entropy",ent
		
		#kpca = KernelPCA(kernel="linear", fit_inverse_transform=True, gamma=10,n_components=pca_components )
		#X_kpca = kpca.fit_transform(X)
		X_kpca = X
		
		#fname = "model/kpca"+ str(ind) +".kpca"
		#joblib.dump(kpca, fname)
		
		fname = "model/kpca"+ str(ind) +".normalizer"
		joblib.dump(normalizer, fname)
		
		clf.fit(X_kpca, y)
		fname = "model/elm"+ str(ind) +".clf"
		joblib.dump(clf, fname)
		
		X = normalizer.transform(X_test)
		#X_kpatest = kpca.transform(X_test);
		X_kpatest = X_test;
		
		print X_test.shape,X.shape,X_kpca.shape,X_kpatest.shape
		y_pred = clf.predict(X_kpatest)
		score = accuracy_score(y_test, y_pred)
		print score
		cm = confusion_matrix(y_test, y_pred)
		print(cm)
		misclass_cost = 0.0;
		for ind_score in range(y_test.shape[0]):
			if y_test[ind_score] != y_pred[ind_score]:
				if y_test[ind_score] == 0:
					misclass_cost += 1/3883370
				elif y_test[ind_score] == 1:
					misclass_cost += 1/41102
				elif y_test[ind_score] == 2:
					misclass_cost += 1/1126
				elif y_test[ind_score] == 3:
					misclass_cost += 1/52
					print ind_score,y_test[ind_score],y_pred[ind_score],misclass_cost
				elif y_test[ind_score] == 4:
					misclass_cost += 1/972781

		fout.write(str(ind) + "\t" + str(score) + "\t" + str(ent) + "\t" + str(misclass_cost) + "\n");
		fout.flush();
	fout.close();
		
		#print X_kpca
		#plt.scatter(X_kpca[:,0], X_kpca[:,1],c=y, marker="o",s=80)
		#plt.show()

		#np.savetxt('kpca.txt',X_kpca, fmt ='%10.5f', delimiter=',')
def predictData():
	sql = "select case label WHEN 'back' then 0 WHEN 'land' then 0 WHEN 'neptune' then 0 WHEN 'pod' then 0 WHEN 'smurf' then 0 WHEN 'teardrop' then 0 WHEN 'ipsweep' then 1 WHEN 'nmap' then 1 WHEN 'portsweep' then 1 WHEN 'satan' then 1 WHEN 'ftp_write' then 2 WHEN 'guess_passwd' then 2 WHEN 'imap' then 2 WHEN 'multihop' then 2 WHEN 'phf' then 2 WHEN 'spy' then 2 WHEN 'warezclient' then 2 WHEN 'warezmaster' then 2 WHEN 'buffer_overflow' then 3 WHEN 'loadmodule' then 3 WHEN 'perl' then 3 WHEN 'rootkit' then 3 when 'normal' then 4  end as label1 ,duration,src_bytes,dst_bytes,wrong_fragment,urgent,hot,num_failed_logins,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,countx,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate from transaction1 where connection in $ids$ or label in( 'buffer_overflow','warezmaster','guess_passwd','loadmodule','warezclient','multihop')";
	ids = [ random.randrange(512,4898942) for i in range(50000)]
	ids = str(ids).replace('[','(').replace(']',')')
		
	rec = engine.execute(sql.replace("$ids$",ids))
	bagsize = 70
	b = open('pred.csv', 'wb')
	w = csv.writer(b)
	for row in rec:
		w.writerow(row)
	b.close();
	D = np.loadtxt('pred.csv', delimiter=',')
	X = D[:,1:38]
	y = D[:,0];
	D = None;
	
	normalizer = sklearn.preprocessing.Normalizer().fit(X)
	X = normalizer.transform(X)
	y_pred_all = np.array([]);
	ind_pnt = 0;
	for row_ind in range(X.shape[0]):
		results = {'0':0,'1':0,'2':0,'3':0,'4':0}
		x_vec = X[row_ind,:]
		ind_pnt += 1;
		if ind_pnt%10 == 0:
			print ind_pnt,datetime.now()
		for ind in range(bagsize):
			#fname = "model/kpca"+ str(ind) +".kpca"
			#kpca = joblib.load(fname)
			#X = kpca.fit_transform(X)
			
			fname = "model/elm"+ str(ind) +".clf"
			clf = joblib.load(fname)
			
			#fname = "model/kpca"+ str(ind) +".normalizer"
			#normalizer = joblib.load(fname)
			
			y_pred = clf.predict(x_vec)
			results[str(int(y_pred))] = results[str(int(y_pred))] + 1;
		sorted_x = sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)
		y_pred_all = np.append(y_pred_all,float(sorted_x[0][0]))
	cm = confusion_matrix(y, y_pred_all)
	print(cm)

if __name__ == '__main__':
	trainModel();
	#predictData();