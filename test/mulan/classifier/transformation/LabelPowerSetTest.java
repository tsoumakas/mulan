package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class LabelPowerSetTest extends TransformationBasedMultiLabelLearnerTest {
	
	@Override
	public void setUp(){
		Classifier baseClassifier = new NaiveBayes();
		learner = new  LabelPowerset(baseClassifier);
	}

}
