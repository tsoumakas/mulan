package mulan.classifier.lazy;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

public class MLkNNTest extends MultiLabelLearnerTestBase {
	
	private static final int DEFAULT_numOfNeighbors = 10; 
	private static final double DEFAULT_smooth = 1.0;
	private static boolean DEFAULT_dontNormalize = true;
	
	private MLkNN learner;
	
	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	public void setUp(){
		learner = new MLkNN();
	}

	@Test
	public void testTestDefaultParameters(){
		Assert.assertEquals(DEFAULT_numOfNeighbors, learner.numOfNeighbors);
		Assert.assertEquals(DEFAULT_smooth, learner.smooth);
		Assert.assertEquals(DEFAULT_dontNormalize, learner.dontNormalize);
		
		// common tests
		Assert.assertTrue(learner.isUpdatable());
	}

}
