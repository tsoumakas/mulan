package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class PPTTest extends TransformationBasedMultiLabelLearnerTest {

	@Override
	public void setUp(){
		Classifier baseClassifier = new NaiveBayes();
		learner = new  PPT(baseClassifier, 2);
	}

}
