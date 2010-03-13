package mulan.classifier.transformation;

import weka.classifiers.trees.J48;

public class LabelPowerSetTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() {
        learner = new LabelPowerset(new J48());
    }
}
