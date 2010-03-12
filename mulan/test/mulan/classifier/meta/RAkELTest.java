package mulan.classifier.meta;

import mulan.classifier.transformation.LabelPowerset;
import weka.classifiers.trees.J48;

public class RAkELTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new RAkEL(new LabelPowerset(new J48()));
	}

}
