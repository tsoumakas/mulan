package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class IncludeLabelsClassifierTest extends
		TransformationBasedMultiLabelLearnerTest {

	@Override
	public void setUp() {
		Classifier baseClassifier = new NaiveBayes();
		learner = new  IncludeLabelsClassifier(baseClassifier);
	}

}
