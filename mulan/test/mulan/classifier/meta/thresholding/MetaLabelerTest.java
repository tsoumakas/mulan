package mulan.classifier.meta.thresholding;

import mulan.classifier.meta.MultiLabelMetaLearnerTest;
import mulan.classifier.transformation.CalibratedLabelRanking;
import weka.classifiers.trees.J48;

public class MetaLabelerTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new MetaLabeler(new CalibratedLabelRanking(new J48()), new J48(), "Content-Based", "Nominal-Class");
    }
}

