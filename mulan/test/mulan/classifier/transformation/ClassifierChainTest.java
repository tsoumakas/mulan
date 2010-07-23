package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class ClassifierChainTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new ClassifierChain(baseClassifier);
    }
}
