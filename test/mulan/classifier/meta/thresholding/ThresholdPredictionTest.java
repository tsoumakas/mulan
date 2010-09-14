package mulan.classifier.meta.thresholding;

import mulan.classifier.meta.MultiLabelMetaLearnerTest;
import mulan.classifier.transformation.CalibratedLabelRanking;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.J48;

public class ThresholdPredictionTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new ThresholdPrediction(new CalibratedLabelRanking(new J48()), new M5P(), "Content-Based", 3);
    }
}

