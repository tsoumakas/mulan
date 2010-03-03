package mulan.classifier.lazy;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

public class BRkNNTest extends MultiLabelLearnerTestBase {

	private static final int DEFAULT_numOfNeighbors = 10; 
	private static boolean DEFAULT_dontNormalize = false;
	
	private BRkNN learner;
	
	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	public void setUp(){
		learner = new BRkNN(DEFAULT_numOfNeighbors);
	}

	@Test
	public void testTestDefaultParameters(){
		Assert.assertEquals(DEFAULT_numOfNeighbors, learner.numOfNeighbors);
		Assert.assertEquals(DEFAULT_dontNormalize, learner.dontNormalize);
		
		// common tests
		Assert.assertTrue(learner.isUpdatable());
	}

}
