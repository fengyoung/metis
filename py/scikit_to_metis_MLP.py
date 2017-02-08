#! /data1/fengyoung/anaconda2/bin/python

import MetisParams
from sklearn import neural_network 
import sys
import datetime


def saveModelAsMetisFmt(clf, model_file, learn_params, arch_params_mlp):
	fp = open(model_file, "w")
	fp.write("** Multiple Layers Perceptron **\n")
	fp.write("\n")

	learn_params.printToFile(fp)
	arch_params_mlp.printToFile(fp)

	hl = len(arch_params_mlp._hiddens)
	for h in range(hl+1):
		fp.write("@weight_" + str(h) + "\n")
		if h == 0: 
			for i in range(arch_params_mlp._input): 
				fp.write(reduce(lambda x, y: str(x) + "," + str(y), clf.coefs_[h][i]) + "\n")
			fp.write(reduce(lambda x, y: str(x) + "," + str(y), clf.intercepts_[h]) + "\n")
			fp.write("\n")
		elif h < hl:
			for i in range(arch_params_mlp._hiddens[h-1]):	
				fp.write(reduce(lambda x, y: str(x) + "," + str(y), clf.coefs_[h][i]) + "\n")
			fp.write(reduce(lambda x, y: str(x) + "," + str(y), clf.intercepts_[h]) + "\n")
			fp.write("\n")
		else:
			for i in range(arch_params_mlp._hiddens[h-1]):	
				fp.write(str(clf.coefs_[h][i][0]) + "," + str(0.0 - clf.coefs_[h][i][0]) + "\n")
			fp.write(str(clf.intercepts_[h][0]) + "," + str(0.0 - clf.intercepts_[h][0]) + "\n")
			fp.write("\n")
	fp.close()


if __name__ == "__main__":
	if len(sys.argv) != 5:
		print "USAGE: %s <targets> <config_file> <patts_file> <model_file>" % sys.argv[0] 
		print "PARAMS:"
		print "  targets - target labels which are as positive, in \"s0,s1,s2,...\" format"
		print "  config_file - config file of learning & architecture parameters"
		print "  patts_file - training patterns file in metis format"
		print "  model_file - out model file as \"Perceptron\" in metis format"
		print ""
		exit(-1)
	
	learn_params = MetisParams.LearnParams()
	arch_params_mlp = MetisParams.ArchParams_MLP()

	targets = map(lambda x: int(x), sys.argv[1].split(","))
	learn_params.readFromConfig(sys.argv[2])	
	arch_params_mlp.readFromConfig(sys.argv[2])	
	
	clf = neural_network.MLPClassifier(hidden_layer_sizes = tuple(arch_params_mlp._hiddens), activation = arch_params_mlp.hiddenActivation(), alpha = learn_params._alpha, batch_size = learn_params._mini_batch, max_iter = learn_params._max_iter, tol = learn_params._epsilon, learning_rate_init = learn_params._learning_rate_init, momentum = learn_params._mom_att, early_stopping = True, epsilon = learn_params._epsilon)

	# read patterns from file	
	time0 = datetime.datetime.now()
	X_train, Y_train = MetisParams.readPatts(sys.argv[3], targets)
	arch_params_mlp._input = len(X_train[0]) 
	arch_params_mlp._output = 2 
	print "Load %d patterns from %s" % (len(Y_train), sys.argv[3])
	
	# train model
	time1 = datetime.datetime.now()
	clf.fit(X_train, Y_train)
	print "Training Completed, number of iterations is %d" % clf.n_iter_

	# save model
	time2 = datetime.datetime.now()
	saveModelAsMetisFmt(clf, sys.argv[4], learn_params, arch_params_mlp)
	print "Model has been saved to %s" % sys.argv[4]

	time3 = datetime.datetime.now()
	print "Time cost: loading(%s) - training(%s) - saving(%s)" % (time1 - time0, time2 - time1, time3 - time2)	

	exit(0)

