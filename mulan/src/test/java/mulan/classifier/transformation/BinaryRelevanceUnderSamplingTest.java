package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class BinaryRelevanceUnderSamplingTest extends TransformationBasedMultiLabelLearnerTest{
	
    @Override
    public void setUp() {
        Classifier baseClassifier = new J48();
        learner = new BinaryRelevanceUnderSampling(baseClassifier);
    }
}
