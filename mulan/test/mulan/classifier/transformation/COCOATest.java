package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class COCOATest extends TransformationBasedMultiLabelLearnerTest {
	@Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new COCOA(baseClassifier,10);
    }
}
