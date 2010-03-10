package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class BinaryRelevanceTest extends TransformationBasedMultiLabelLearnerTest {

	@Override
	public void setUp(){
		Classifier baseClassifier = new NaiveBayes();
		learner = new  BinaryRelevance(baseClassifier);
	}
	
}
