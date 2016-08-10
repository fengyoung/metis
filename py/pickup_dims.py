#! /usr/bin/python
# encoding=utf8

import sys


def readIndex2Ig(ig_file):
	fp = open(ig_file, "r")
	idx_ig_list = []	
	line = fp.readline()
	while(line):
		ss = line.rstrip().split(" ")
		idx_ig_list.append([int(ss[1]), float(ss[2])])
		line = fp.readline()
	fp.close()
	return idx_ig_list


def dimPickup(vec_ss, idx_igs, topn):
	dims = topn
	if dims > len(idx_igs):
		dims = len(idx_igs)
	vec_new_ss = []
	for k in range(dims):
		vec_new_ss.append(vec_ss[idx_igs[k][0]])
	return vec_new_ss



if __name__ == "__main__":
	if len(sys.argv) != 5:
		print "Usage: %s <input patterns file> <ig_file> <top n> <output patterns file>" % sys.argv[0]
		exit(-1)

	topn = int(sys.argv[3])
	idx_igs = readIndex2Ig(sys.argv[2])

	print idx_igs

	
	fp_in = open(sys.argv[1], "r")
	fp_out = open(sys.argv[4], "w")

	line = fp_in.readline()
	while(line):
		label_vec = line.rstrip().split(";")
		vec_ss = label_vec[1].split(",")		
		new_vec_ss = dimPickup(vec_ss, idx_igs, topn)
		fp_out.write(label_vec[0] + ";" + reduce(lambda x, y: x + "," + y, new_vec_ss) + "\n")
		line = fp_in.readline()

	fp_in.close()
	fp_out.close()
	
	exit(0)
 
