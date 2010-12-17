package mulan.classifier.meta.thresholding;

import mulan.classifier.meta.MultiLabelMetaLearnerTest;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.evaluation.measure.HammingLoss;
import weka.classifiers.trees.J48;

public class OneThresholdTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new OneThreshold(new CalibratedLabelRanking(new J48()), new HammingLoss(), 3);
    }
}

