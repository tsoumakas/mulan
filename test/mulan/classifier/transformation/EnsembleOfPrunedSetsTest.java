package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class EnsembleOfPrunedSetsTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new EnsembleOfPrunedSets(63, 10, 0.5, 2, PrunedSets.Strategy.A, 3, baseClassifier);
    }
}
