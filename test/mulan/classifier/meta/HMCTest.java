package mulan.classifier.meta;

import mulan.classifier.transformation.BinaryRelevance;
import weka.classifiers.trees.J48;

public class HMCTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new HMC(new BinaryRelevance(new J48()));
    }
}
