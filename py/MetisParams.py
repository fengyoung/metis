#! /data1/fengyoung/anaconda2/bin/python


class LearnParams:
	def __init__(self): 
		self._regula = 'l2'
		self._optim = 'NAG'
		self._mini_batch = 200
		self._max_iter = 100
		self._early_stop = 10
		self._learning_rate_init = 0.001
		self._mom_att = 0.9
		self._alpha = 0.0001
		self._epsilon = 0.01

	def readFromConfig(self, config_file): 
		fp = open(config_file)
		line = fp.readline()
		while(line):
			if line[0] == '#':
				line = fp.readline()
				continue
			if len(line.rstrip()) == 0:
				line = fp.readline()
				continue
			kv = map(lambda x: x.rstrip().lstrip(), line.split("="))
			if kv[0] == "Regula":
				self._regula = kv[1].lower()
			elif kv[0] == "Optim":
				self._optim = kv[1]
			elif kv[0] == "MiniBatch":
				self._mini_batch = int(kv[1])
			elif kv[0] == "MaxIter":
				self._max_iter = int(kv[1])
			elif kv[0] == "EarlyStop":
				self._early_stop = int(kv[1])
			elif kv[0] == "LearningRateInit":
				self._learning_rate_init = float(kv[1])
			elif kv[0] == "MomAtt":
				self._mom_att = float(kv[1])
			elif kv[0] == "Alpha":
				self._alpha = float(kv[1])
			elif kv[0] == "Epsilon":
				self._epsilon = float(kv[1])
			line = fp.readline()
		fp.close()

	def printToFile(self, fp_out):
		fp_out.write("@learning_params\n")
		fp_out.write("Regula:" + self._regula.upper() + "\n")
		fp_out.write("Optim:" + self._optim + "\n")
		fp_out.write("MiniBatch:" + str(self._mini_batch) + "\n")
		fp_out.write("MaxIter:" + str(self._max_iter) + "\n")
		fp_out.write("EarlyStop:" + str(self._early_stop) + "\n")
		fp_out.write("LearningRateInit:" + str(self._learning_rate_init) + "\n")
		fp_out.write("MomAtt:" + str(self._mom_att) + "\n")
		fp_out.write("Alpha:" + str(self._alpha) + "\n")
		fp_out.write("Epsilon:" + str(self._epsilon) + "\n")
		fp_out.write("\n")


class ArchParams:
	def __init__(self):
		self._input = 0 
		self._output = 0 

	def printToFile(self, fp_out):
		fp_out.write("@architecture_params\n")
		fp_out.write("Input:" + str(self._input) + "\n")
		fp_out.write("Output:" + str(self._output) + "\n")
		fp_out.write("\n")


class ArchParams_MLP:
	def __init__(self):
		self._input = 0 
		self._output = 0 
		self._hiddens = []
		self._act_hidden = 'relu'

	def printToFile(self, fp_out):
		fp_out.write("@architecture_params\n")
		fp_out.write("Input:" + str(self._input) + "\n")
		if len(self._hiddens) == 1:
			fp_out.write("Hiddens:" + str(self._hiddens[0]) + "\n")
		elif len(self._hiddens) > 1:
			fp_out.write("Hiddens:" + reduce(lambda x, y: str(x) + ',' + str(y), self._hiddens) + "\n")
		else:
			fp_out.write("Hiddens:null\n")
		fp_out.write("Output:" + str(self._output) + "\n")
		fp_out.write("ActHidden:" + str(self._act_hidden) + "\n")
		fp_out.write("\n")
	
	def readFromConfig(self, config_file): 
		fp = open(config_file)
		line = fp.readline()
		while(line):
			if line[0] == '#':
				line = fp.readline()
				continue
			if len(line.rstrip()) == 0:
				line = fp.readline()
				continue
			kv = map(lambda x: x.rstrip().lstrip(), line.split("="))
			if kv[0] == "Hiddens":
				self._hiddens = map(lambda x: int(x), kv[1].split(','))
			elif kv[0] == "ActHidden:":
				self._act_hidden = kv[1]
			line = fp.readline()
		fp.close()

	def hiddenActivation(self, scikit_fmt = True):
		if scikit_fmt == True:
			if self._act_hidden == 'linear':
				return 'identity'
			elif self._act_hidden == 'sigmoid':
				return 'logistic'
			elif self._act_hidden == 'tanh':
				return 'tanh'
			elif self._act_hidden == 'relu':
				return 'relu'
			else:
				return 'none' 
		else:
			return self._act_hidden


def readPatts(patts_file, targets):
	X = []
	Y = []
	fp = open(patts_file, "r")
	line = fp.readline()
	while line:
		patt = line.rstrip().split(";")
		ys = patt[0].split(",")
		y = 0
		for t in targets: 
			if int(ys[t]) == 1:
				y = 1
		x = map(lambda x: float(x), patt[1].split(",")) 
		X.append(x)
		Y.append(y)	
		line = fp.readline()
	fp.close()
	return (X, Y)


if __name__ == "__main__":
	learn_params = LearnParams()
	arch_params = ArchParams() 
	arch_params_mlp = ArchParams_MLP() 

	fp = open("out.txt", "w")
	learn_params.printToFile(fp)
	arch_params.printToFile(fp)
	arch_params_mlp.printToFile(fp)

	learn_params.readFromConfig("../config/mlp_params.conf")
	learn_params.printToFile(fp)
	arch_params_mlp.readFromConfig("../config/mlp_params.conf")
	arch_params_mlp.printToFile(fp)

	fp.close()

	exit(0)

