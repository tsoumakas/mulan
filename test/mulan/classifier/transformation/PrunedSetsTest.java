package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class PrunedSetsTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new PrunedSets(baseClassifier, 2, PrunedSets.Strategy.A, 3);
    }
}
