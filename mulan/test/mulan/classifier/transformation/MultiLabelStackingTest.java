package mulan.classifier.transformation;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;

public class MultiLabelStackingTest extends TransformationBasedMultiLabelLearnerTest {

    @Override
    public void setUp() throws Exception {
        Classifier baseClassifier = new IBk();
        Classifier metaClassifier = new Logistic();
        learner = new MultiLabelStacking(baseClassifier, metaClassifier);
    }
}
