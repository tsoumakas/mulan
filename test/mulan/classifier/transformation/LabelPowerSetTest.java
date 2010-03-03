package mulan.classifier.transformation;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

import org.junit.Before;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class LabelPowerSetTest extends MultiLabelLearnerTestBase {
	
	private LabelPowerset learner;
	
	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	public void setUp(){
		Classifier baseClassifier = new NaiveBayes();
		learner = new  LabelPowerset(baseClassifier);
	}

}
