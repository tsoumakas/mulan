package mulan.classifier.transformation;

import mulan.transformations.multiclass.Copy;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class MultiClassLearnerTest extends
		TransformationBasedMultiLabelLearnerTest {

	@Override
	public void setUp() {
		Classifier baseClassifier = new NaiveBayes();
		Copy cptransformation = new Copy();
		learner = new  MultiClassLearner(baseClassifier, cptransformation);
		// TO DO: test with other transformations
	}

}
