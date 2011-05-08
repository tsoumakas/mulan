package mulan.classifier.lazy;

import junit.framework.Assert;

import org.junit.Test;

public class BRkNNTest extends MultiLabelKNNTest {

	private static final int DEFAULT_numOfNeighbors = 10;

	@Override
	public void setUp() {
		learner = new BRkNN(DEFAULT_numOfNeighbors);
	}

	@Test
	public void testTestDefaultParameters() {
		Assert.assertEquals(DEFAULT_numOfNeighbors, learner.numOfNeighbors);

		// common tests
		Assert.assertTrue(learner.isUpdatable());
	}
}
