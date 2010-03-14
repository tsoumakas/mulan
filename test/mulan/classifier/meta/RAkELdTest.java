package mulan.classifier.meta;

import mulan.classifier.transformation.LabelPowerset;
import weka.classifiers.trees.J48;

public class RAkELdTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new RAkELd(new LabelPowerset(new J48()));
    }
}
