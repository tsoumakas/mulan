package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class EnsembleOfClassifierChainsTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new EnsembleOfClassifierChains(baseClassifier, 10, false, false);
    }
}
