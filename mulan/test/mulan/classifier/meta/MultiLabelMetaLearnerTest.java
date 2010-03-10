package mulan.classifier.meta;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

import org.junit.Before;
import org.junit.Ignore;

@Ignore
public abstract class MultiLabelMetaLearnerTest extends MultiLabelLearnerTestBase {
	
	protected MultiLabelMetaLearner learner;

	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	abstract public void setUp () throws Exception;

}
