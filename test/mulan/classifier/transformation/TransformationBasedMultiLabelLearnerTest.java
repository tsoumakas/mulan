package mulan.classifier.transformation;

import org.junit.Before;
import org.junit.Ignore;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

@Ignore
public abstract class TransformationBasedMultiLabelLearnerTest extends
		MultiLabelLearnerTestBase {

	protected TransformationBasedMultiLabelLearner learner;
	
	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	abstract public void setUp () throws Exception;
}
