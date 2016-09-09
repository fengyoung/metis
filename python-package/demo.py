# -*- coding utf-8 -*-
import sys
import metis_predict


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage: %s model test_data' % sys.argv[0]
        sys.exit(-1)

    model_file = sys.argv[1]
    model = metis_predict.Model_LoadModel(model_file)

    test_data_file = sys.argv[2]
    fd = open(test_data_file, 'r')
    for line in fd:
        arr1 = line.strip().split(';')
        label = arr1[0]
        feat_str = arr1[1]
        feat = feat_str.split(',')

        pv = metis_predict.PairVector()
        for idx, value in enumerate(feat):
            pv.push_back((int(idx), float(value)))

        pred_score = model.Predict(pv)
        print "predict score", pred_score
    fd.close()
