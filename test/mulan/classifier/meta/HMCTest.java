package mulan.classifier.meta;

import mulan.classifier.lazy.BRkNN;

public class HMCTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new  HMC(new BRkNN(10));
	}

}
