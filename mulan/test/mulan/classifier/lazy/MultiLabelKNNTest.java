package mulan.classifier.lazy;

import org.junit.Before;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

public abstract class MultiLabelKNNTest extends MultiLabelLearnerTestBase {
	
	protected MultiLabelKNN learner;

	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}

	@Before
	abstract public void setUp();
}
