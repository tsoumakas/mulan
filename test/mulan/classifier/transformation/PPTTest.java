package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class PPTTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new PPT(baseClassifier, 2, PPT.Strategy.NO_INFORMATION_LOSS);
    }
}
