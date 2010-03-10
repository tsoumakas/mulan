package meta;

import org.junit.Before;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;
import mulan.classifier.meta.MultiLabelMetaLearner;

public abstract class MultiLabelMetaLearnerTest extends MultiLabelLearnerTestBase {
	
	protected MultiLabelMetaLearner learner;

	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	abstract public void setUp () throws Exception;

}
