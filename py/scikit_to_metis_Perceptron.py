#! /data1/fengyoung/anaconda2/bin/python

import MetisParams
from sklearn import linear_model
import sys
import datetime


def saveModelAsMetisFmt(clf, model_file, learn_params, arch_params):
	fp = open(model_file, "w")
	fp.write("** Perceptron **\n")
	fp.write("\n")

	learn_params.printToFile(fp)
	arch_params.printToFile(fp)

	fp.write("@weight\n")
	for i in range(arch_params._input):
		fp.write(str(clf.coef_[0][i]) + "," + str(0.0 - clf.coef_[0][i]) + "\n")
	fp.write(str(clf.intercept_[0]) + "," + str(0.0 - clf.intercept_[0]) + "\n")
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
	arch_params = MetisParams.ArchParams()
	
	targets = map(lambda x: int(x), sys.argv[1].split(","))
	learn_params.readFromConfig(sys.argv[2])	

	clf = linear_model.LogisticRegression(penalty = learn_params._regula, C = learn_params._alpha, max_iter = learn_params._max_iter, tol = learn_params._epsilon)
	
	# read patterns from file	
	time0 = datetime.datetime.now()
	X_train, Y_train = MetisParams.readPatts(sys.argv[3], targets)
	arch_params._input = len(X_train[0]) 
	arch_params._output = 2 
	print "Load %d patterns from %s" % (len(Y_train), sys.argv[3])

	# train model
	time1 = datetime.datetime.now()
	clf.fit(X_train, Y_train)
	print "Training Completed, number of iterations is %d" % clf.n_iter_
	
	# save model
	time2 = datetime.datetime.now()
	saveModelAsMetisFmt(clf, sys.argv[4], learn_params, arch_params)
	time3 = datetime.datetime.now()
	print "Model has been saved to %s" % sys.argv[4]

	print "Time cost: loading(%s) - training(%s) - saving(%s)" % (time1 - time0, time2 - time1, time3 - time2)	

	exit(0)

